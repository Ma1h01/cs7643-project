import os
import json
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, get_cosine_schedule_with_warmup
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import your custom dataset
from dataset import EldenRingDataset

# --- DCFT ARCHITECTURE ---

class DCFTLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r=16, kernel_size=4, stride=2, dropout=0.1):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # 1. Freeze original weights
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
            
        # 2. True DCFT Subspace Configuration
        self.r = r
        # We compress down to a micro-dimension so deconv can expand it back to 'r'
        self.compressed_dim = max(1, self.r // stride)
        
        # A: Down-projection (d -> compressed_dim)
        self.down_proj = nn.Linear(self.in_features, self.compressed_dim, bias=False)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # d: 1D Deconvolution layer (compressed_dim -> r)
        self.deconv = nn.ConvTranspose1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=kernel_size // 2 
        )
        
        # B: Up-projection (r -> d)
        deconv_out_len = (self.compressed_dim - 1) * stride - 2 * (kernel_size // 2) + kernel_size
        self.up_proj = nn.Linear(deconv_out_len, self.out_features, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight) 
        nn.init.normal_(self.deconv.weight, std=0.02)

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)
        
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features) 
        
        # Step 1: Micro-Subspace Projection + Dropout
        subspace_x = self.down_proj(x_flat) 
        subspace_x = self.dropout(subspace_x)
        
        # Step 2: 1D Deconvolution back to rank 'r'
        subspace_x = subspace_x.unsqueeze(1)
        deconv_x = self.deconv(subspace_x).squeeze(1) 
        
        # Step 3: Reconstruction and Residual Sum
        dcft_out = self.up_proj(deconv_x)
        dcft_out = dcft_out.view(*orig_shape[:-1], self.out_features)
        
        return base_out + dcft_out

def inject_dcft(module, target_modules=["to_q", "to_k", "to_v", "to_out.0"], r=16, k=4, s=2, dropout=0.1):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
            setattr(module, name, DCFTLinear(child, r=r, kernel_size=k, stride=s, dropout=dropout))
        else:
            inject_dcft(child, target_modules, r, k, s, dropout)

def parse_args():
    parser = argparse.ArgumentParser(description="Train DCFT for Stable Diffusion")
    parser.add_argument("--r", type=int, default=16) # ADD THIS: Explicit bottleneck
    parser.add_argument("--kernel_size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2) # We can safely return to stride 2
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="elden_ring_dcft_weights")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "sd-legacy/stable-diffusion-v1-5"

    # 1. Load Components
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 2. Inject DCFT
    print(f"Injecting DCFT (k={args.kernel_size}, s={args.stride}, dropout={args.dropout})...")
    inject_dcft(unet, r=args.r, k=args.kernel_size, s=args.stride, dropout=args.dropout)

    unet.to(device)
    
    # 3. Filter Parameters
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total DCFT Trainable Parameters: {num_trainable:,}")

    # 4. Optimizer & Data
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    train_dataset = EldenRingDataset("combined_data", "train.jsonl", tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    total_steps = (len(train_dataloader) // args.grad_acc_steps) * args.epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    scaler = GradScaler('cuda')

    # 5. Training Loop
    unet.train()
    losses = []

    for epoch in range(args.epoch):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float16)).latent_dist.sample() * 0.18215
                encoder_states = text_encoder(batch["input_ids"].to(device))[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with autocast(device_type="cuda"):
                noise_pred = unet(noisy_latents, timesteps, encoder_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float()) / args.grad_acc_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_acc_steps == 0:
                # Gradient Clipping Added Here
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                losses.append(loss.item() * args.grad_acc_steps)
                progress_bar.set_postfix({"loss": f"{losses[-1]:.4f}"})

    # 6. Save State
    os.makedirs(args.out_dir, exist_ok=True)
    dcft_state = {k: v for k, v in unet.state_dict().items() if "down_proj" in k or "up_proj" in k or "deconv" in k}
    torch.save(dcft_state, os.path.join(args.out_dir, "dcft_weights.pt"))
    print(f"DCFT Weights saved to {args.out_dir}")

if __name__ == "__main__":
    main()