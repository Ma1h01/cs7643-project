import os
import json
import argparse
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from diffusers import StableDiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm.auto import tqdm

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

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on {device}...")

    # 1. Initialize Metrics
    # FID and KID expect features from the InceptionV3 network
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device) # subset_size adjusted for small datasets
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    # 2. Load the Base Model and Inject PEFT Weights
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True) # Hide individual generation bars for cleaner logs
    
    # Load the specific PEFT weights your teammate passes in
    print(f"Loading PEFT weights from: {args.peft_path}")
    pipe.load_lora_weights(args.peft_path)

    # 3. Load Test Data
    test_data = load_test_data(args.test_jsonl, args.data_dir)
    print(f"Loaded {len(test_data)} test samples.")

    # 4. Generation and Metric Update Loop
    print("Generating images and updating metrics...")
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

        # Update Distribution Metrics (FID/KID)
        # real=True for ground truth, real=False for generated
        fid.update(real_img_tensor, real=True)
        fid.update(fake_img_tensor, real=False)
        
        kid.update(real_img_tensor, real=True)
        kid.update(fake_img_tensor, real=False)

        # Update Alignment Metric (CLIPScore)
        clip_score.update(fake_img_tensor.squeeze(0), prompt)

    # 5. Compute Final Scores
    print("\nComputing final scores...")
    final_fid = fid.compute().item()
    final_kid = kid.compute() # Returns a tuple (kid_mean, kid_std)
    final_clip = clip_score.compute().item()

    # 6. Output Results
    results = {
        "Model": args.peft_path,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SD models.")
    parser.add_argument("--peft_path", type=str, required=True, help="Path to the saved PEFT weights directory.")
    parser.add_argument("--test_jsonl", type=str, default="test.jsonl", help="Path to the test JSONL file.")
    parser.add_argument("--data_dir", type=str, default="combined_data", help="Directory containing the images.")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="File to save the metrics.")
    
    args = parser.parse_args()
    main(args)