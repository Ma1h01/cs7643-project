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
from peft import PeftModel

def load_test_data(jsonl_file, img_dir):
    """Loads the test prompts and corresponding real image paths."""
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "prompt": item["text"],
                "real_img_path": os.path.join(img_dir, item["file_name"])
            })
    return data

def preprocess_image(image, size=512):
    """Resizes and converts PIL Image to the format expected by torchmetrics (uint8 tensor)."""
    image = image.resize((size, size))
    image = TF.pil_to_tensor(image) # Returns uint8 tensor in [0, 255]
    return image.unsqueeze(0) # Add batch dimension

def preprocess_image_for_clip(image, size=512):
    """Resizes and converts PIL Image to tensor format expected by CLIPScore."""
    image = image.resize((size, size))
    image_tensor = TF.pil_to_tensor(image).float() / 255.0  # Convert to float [0, 1]
    # CLIPScore expects [C, H, W] format (no batch dimension for single image)
    return image_tensor

def load_peft_model(pipe, peft_path, peft_type):
    """
    Loads PEFT weights into the pipeline's UNet.
    Tries PEFT-native loading first, then falls back to diffusers-native.
    """
    print(f"Loading PEFT type: {peft_type}")

    if peft_type.upper() in ["LORA", "ADALORA", "DORA"]:
        # Check if this is a PEFT-format checkpoint (has adapter_config.json)
        is_peft_format = os.path.exists(os.path.join(peft_path, "adapter_config.json"))

        if is_peft_format:
            # --- Path A: PEFT library trained ---
            pipe.unet = PeftModel.from_pretrained(pipe.unet, peft_path)
            print("Loaded via PeftModel.from_pretrained() [PEFT format]")
        else:
            # --- Path B: Diffusers-native LoRA ---
            pipe.load_lora_weights(peft_path)
            print("Loaded via pipe.load_lora_weights() [diffusers format]")

    return pipe

def count_lora_params(unet):
    """
    Correctly counts LoRA parameters regardless of loading method.
    - PEFT format: lora_A / lora_B appear in named_parameters()
    - Diffusers native: LoRA weights live in attn_processors
    """
    lora_params = 0
    base_params = 0

    # Method 1: PEFT-style (lora_A, lora_B in named_parameters)
    for name, param in unet.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params += param.numel()
        else:
            base_params += param.numel()

    # Method 2: Diffusers-native (weights live in attn_processors)
    # Only use this if Method 1 found nothing
    if lora_params == 0:
        for name, processor in unet.attn_processors.items():
            for attr_name in dir(processor):
                if "lora" in attr_name.lower():
                    attr = getattr(processor, attr_name)
                    if hasattr(attr, "weight"):
                        lora_params += attr.weight.numel()

        # Recount base params as total - lora
        total_from_named = sum(p.numel() for p in unet.parameters())
        base_params = total_from_named - lora_params

    total_params = base_params + lora_params
    return base_params, lora_params, total_params

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on {device}...")

    # 1. Initialize Metrics
    # FID and KID expect features from the InceptionV3 network
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device) # subset_size adjusted for small datasets
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    # --- HOTFIX FOR TRANSFORMERS / TORCHMETRICS MISMATCH ---
    def extract_tensor(features):
        if isinstance(features, torch.Tensor):
            return features
        # Extract the tensor from the BaseModelOutput object
        if hasattr(features, 'image_embeds'):
            return features.image_embeds
        if hasattr(features, 'text_embeds'):
            return features.text_embeds
        if hasattr(features, 'pooler_output'):
            return features.pooler_output
        return features
        
    original_get_image = clip_score.model.get_image_features
    clip_score.model.get_image_features = lambda *args, **kwargs: extract_tensor(original_get_image(*args, **kwargs))
    
    original_get_text = clip_score.model.get_text_features
    clip_score.model.get_text_features = lambda *args, **kwargs: extract_tensor(original_get_text(*args, **kwargs))
    # -------------------------------------------------------

    # 2. Load the Base Model and Inject PEFT Weights
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Load the specific PEFT weights
    print(f"Loading PEFT weights from: {args.peft_path}")
    pipe = load_peft_model(pipe, args.peft_path, args.peft_type)

    # Count LoRA/Adapter parameters in UNet
    base_params, lora_params, total_params = count_lora_params(pipe.unet)
    lora_pct = round(100 * lora_params / total_params, 4) if total_params > 0 else 0.0

    print(f"\nUNet Parameter Breakdown:")
    print(f"  Base params:  {base_params:,}")
    print(f"  LoRA params:  {lora_params:,}")
    print(f"  Total params: {total_params:,}")
    print(f"  LoRA %:       {lora_pct}%")

    if lora_params == 0:
        print("⚠ WARNING: No LoRA parameters detected — check your checkpoint format.")
    else:
        print(f"✓ Found {lora_params:,} LoRA parameters")

    # 3. Load Test Data
    test_data = load_test_data(args.test_jsonl, args.data_dir)
    print(f"Loaded {len(test_data)} test samples.")

    # 4. Generation and Metric Update Loop
    os.makedirs(args.samples_dir, exist_ok=True)

    print("Generating images and updating metrics...")
    save_count = 0
    for item in tqdm(test_data, desc="Evaluating"):
        prompt = item["prompt"]
        
        # Load and preprocess real image
        real_img_pil = Image.open(item["real_img_path"]).convert("RGB")
        real_img_tensor = preprocess_image(real_img_pil).to(device)

        # Generate fake image
        # Using a fixed seed for reproducibility across different PEFT evaluations
        generator = torch.Generator(device).manual_seed(42) 
        fake_img_pil = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
        fake_img_tensor = preprocess_image(fake_img_pil).to(device)
        
        # Save side-by-side comparison for sample images
        comparison = Image.new("RGB", (1024, 512))
        comparison.paste(real_img_pil.resize((512, 512)), (0, 0))
        comparison.paste(fake_img_pil.resize((512, 512)), (512, 0))
        comparison.save(os.path.join(args.samples_dir, f"sample_{save_count:03d}.png"))
        save_count += 1

        # Update Distribution Metrics (FID/KID)
        # real=True for ground truth, real=False for generated
        fid.update(real_img_tensor, real=True)
        fid.update(fake_img_tensor, real=False)
        
        kid.update(real_img_tensor, real=True)
        kid.update(fake_img_tensor, real=False)

        # Update Alignment Metric (CLIPScore) - use tensor format [C, H, W]
        clip_score.update(fake_img_tensor.squeeze(0), prompt)

    # 5. Compute Final Scores
    print("\nComputing final scores...")
    final_fid = fid.compute().item()
    final_kid = kid.compute() # Returns a tuple (kid_mean, kid_std)
    final_clip = clip_score.compute().item()

    # 6. Output Results
    results = {
        "Model": args.peft_path,
        "Base Parameters": base_params,
        "LoRA Parameters": lora_params,
        "Total Parameters": total_params,
        "LoRA %": lora_pct,
        "FID (Lower is better)": round(final_fid, 4),
        "KID Mean (Lower is better)": round(final_kid[0].item(), 4),
        "KID Std": round(final_kid[1].item(), 4),
        "CLIPScore (Higher is better)": round(final_clip, 4)
    }

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Save to file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output_file}")

    # 7. Generate Images for Custom Prompts
    custom_prompts = [
        "eldenring gameplay style, third person view, player character fighting a massive dragon boss, dynamic motion, sparks, fire, cinematic camera angle",
        "eldenring gameplay style, boss arena, grotesque humanoid monster, detailed armor, high contrast lighting, epic composition",
        "eldenring gameplay style, open world exploration, distant castle, broken bridges, dead trees, misty environment, wide shot"
    ]
    
    print(f"\nGenerating {len(custom_prompts)} custom Elden Ring style images...")
    for i, prompt in enumerate(custom_prompts, 1):
        print(f"Generating image {i}/{len(custom_prompts)}: {prompt[:60]}...")
        generator = torch.Generator(device).manual_seed(42 + i)  # Different seed for each image
        custom_img = pipe(prompt, num_inference_steps=30, generator=generator, guidance_scale=7.5).images[0]
        output_path = f"{args.samples_dir}/eldenring_custom_{i:02d}.png"
        custom_img.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SD models.")
    parser.add_argument("--peft_path", type=str, required=True, help="Path to the saved PEFT weights directory.")
    parser.add_argument("--peft_type", type=str, choices=["lora", "adalora", "dora", "dcft"],
                       required=True, help="Type of PEFT method used.")
    parser.add_argument("--test_jsonl", type=str, default="test.jsonl", help="Path to the test JSONL file.")
    parser.add_argument("--data_dir", type=str, default="combined_data", help="Directory containing the images.")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="File to save the metrics.")
    parser.add_argument("--samples_dir", type=str, default="eval_samples", help="Directory to save comparison images.")
    
    args = parser.parse_args()
    main(args)