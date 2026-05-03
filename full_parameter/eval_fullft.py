import os
import json
import argparse
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm.auto import tqdm


def load_test_data(jsonl_file, img_dir):
    """Load prompts and corresponding real-image paths from test.jsonl."""
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "prompt": item["text"],
                "real_img_path": os.path.join(img_dir, item["file_name"])
            })
    return data


def preprocess_image(image, size=512):
    """Resize PIL image to uint8 tensor format expected by FID/KID."""
    image = image.resize((size, size))
    image = TF.pil_to_tensor(image)  # uint8 in [0,255], shape [C,H,W]
    return image.unsqueeze(0)        # [1,C,H,W]


def count_unet_params(unet):
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    return trainable_params, total_params


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on {device}...")

    # ---------------------------
    # 1. Metrics
    # ---------------------------
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device)
    # try:
    #     clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    #     use_clip = True
    # except Exception as e:
    #     print(f"Skipping CLIPScore because model loading failed: {e}")
    #     clip_score = None
    #     use_clip = False

    # Hotfix for some transformers / torchmetrics compatibility issues
    def extract_tensor(features):
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "image_embeds"):
            return features.image_embeds
        if hasattr(features, "text_embeds"):
            return features.text_embeds
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        return features

    # original_get_image = clip_score.model.get_image_features
    # clip_score.model.get_image_features = lambda *args, **kwargs: extract_tensor(original_get_image(*args, **kwargs))

    # original_get_text = clip_score.model.get_text_features
    # clip_score.model.get_text_features = lambda *args, **kwargs: extract_tensor(original_get_text(*args, **kwargs))

    # ---------------------------
    # 2. Load FULL fine-tuned model
    # ---------------------------
    print(f"Loading full fine-tuned model from: {args.model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.requires_safety_checker = False

    # Count UNet params
    unet = UNet2DConditionModel.from_pretrained(os.path.join(args.model_path, "unet"))
    trainable_params, total_params = count_unet_params(unet)

    print("\nUNet Parameter Breakdown:")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable %:      {100 * trainable_params / total_params:.4f}%")

    # ---------------------------
    # 3. Load test data
    # ---------------------------
    test_data = load_test_data(args.test_jsonl, args.data_dir)
    print(f"Loaded {len(test_data)} test samples.")

    os.makedirs(args.samples_dir, exist_ok=True)

    # ---------------------------
    # 4. Generate + update metrics
    # ---------------------------
    print("Generating images and updating metrics...")
    save_count = 0

    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        prompt = item["prompt"]

        # Real image
        real_img_pil = Image.open(item["real_img_path"]).convert("RGB")
        real_img_tensor = preprocess_image(real_img_pil, size=args.image_size).to(device)

        # Generated image
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        fake_img_pil = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        ).images[0]
        fake_img_tensor = preprocess_image(fake_img_pil, size=args.image_size).to(device)

        # Save side-by-side comparison
        comparison = Image.new("RGB", (args.image_size * 2, args.image_size))
        comparison.paste(real_img_pil.resize((args.image_size, args.image_size)), (0, 0))
        comparison.paste(fake_img_pil.resize((args.image_size, args.image_size)), (args.image_size, 0))
        comparison.save(os.path.join(args.samples_dir, f"sample_{save_count:03d}.png"))
        save_count += 1

        # FID / KID
        fid.update(real_img_tensor, real=True)
        fid.update(fake_img_tensor, real=False)

        kid.update(real_img_tensor, real=True)
        kid.update(fake_img_tensor, real=False)

        # CLIPScore: generated image vs prompt
        # if use_clip:
        #     clip_score.update(fake_img_tensor.squeeze(0), prompt)

    # ---------------------------
    # 5. Compute final scores
    # ---------------------------
    print("\nComputing final scores...")
    final_fid = fid.compute().item()
    final_kid = kid.compute()  # (mean, std)
    # final_clip = clip_score.compute().item()

    # ---------------------------
    # 6. Save results
    # ---------------------------
    results = {
        "Model": args.model_path,
        "Trainable Parameters (UNet)": trainable_params,
        "Total Parameters (UNet)": total_params,
        "Trainable % (UNet)": round(100 * trainable_params / total_params, 4),
        "FID (Lower is better)": round(final_fid, 4),
        "KID Mean (Lower is better)": round(final_kid[0].item(), 6),
        "KID Std": round(final_kid[1].item(), 6),
        # "CLIPScore (Higher is better)": round(final_clip, 4)
    }

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a full fine-tuned Stable Diffusion model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved full fine-tuned model directory, e.g. /home/.../full_ft_sd15"
    )
    parser.add_argument(
        "--test_jsonl",
        type=str,
        required=True,
        help="Path to test.jsonl"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the real test images referenced by test.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_results_fullft.json",
        help="Path to save JSON results"
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="eval_samples_fullft",
        help="Directory to save real-vs-generated comparison images"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Resize size for metric computation"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps for generation"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducible generation"
    )

    args = parser.parse_args()
    main(args)
