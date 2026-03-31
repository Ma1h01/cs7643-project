import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse

# Import your custom dataset
from dataset import EldenRingDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for Stable Diffusion")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--grad_acc_steps", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for optimizer (default: 1e-8)")
    parser.add_argument("--epoch", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--lora_weights_out", type=str, default="elden_ring_lora_weights", 
                       help="Output directory for LoRA weights (default: elden_ring_lora_weights)")
    parser.add_argument("--lora_train_results_out", type=str, default="lora_train_results.png", 
                       help="Output file for training results plot (default: lora_train_results.png)")
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "sd-legacy/stable-diffusion-v1-5"

# --- 1. Load Standard SD Components ---
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Freeze base models
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# Set text encoder to eval mode for efficiency
text_encoder.eval()

# --- 2. Setup PEFT (LoRA) with improved config ---
lora_config = LoraConfig(
    r=args.lora_r,  # Configurable rank
    lora_alpha=args.lora_alpha,  # Configurable alpha
    # target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"], # Attention + feed-forward (better style/content capture)
    lora_dropout=args.lora_dropout,  # Configurable dropout
)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# --- 3. Setup DataLoaders with smaller batch size ---
train_dataset = EldenRingDataset("combined_data", "train.jsonl", tokenizer)
val_dataset = EldenRingDataset("combined_data", "val.jsonl", tokenizer)

# Configurable batch size and gradient accumulation
batch_size = args.batch_size
gradient_accumulation_steps = args.grad_acc_steps  # Effective batch size = batch_size * grad_acc_steps

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- 4. Setup optimizer and scheduler with improved hyperparams ---
learning_rate = args.lr  # Configurable learning rate
trainable_params = filter(lambda p: p.requires_grad, unet.parameters())
optimizer = torch.optim.AdamW(
    # unet.parameters(), 
    trainable_params,  # Only optimize trainable parameters (this save GPU memory b/c we not loading all non-trainable params into optimizer)
    lr=learning_rate,
    weight_decay=args.weight_decay,  # Configurable weight decay
    betas=(0.9, 0.999),
    eps=args.eps  # Configurable epsilon
)

num_epochs = args.epoch  # Configurable number of epochs 
# warmup_steps = len(train_dataloader) // gradient_accumulation_steps
total_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
warmup_steps = max(100, total_steps // 20) # Warm up 5% of training

# Cosine annealing with warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

scaler = torch.amp.GradScaler('cuda')

# Enhanced tracking for plotting
epoch_train_losses = []
epoch_val_losses = []
step_losses = []
learning_rates = []

# --- 5. Training & Validation Loop ---
global_step = 0

print(f"Training for {num_epochs} epochs")
print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
print(f"Learning rate: {learning_rate}, Warmup steps: {warmup_steps}")

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    
    # --- TRAINING PHASE ---
    unet.train()
    total_train_loss = 0
    accumulated_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        try:
            # Convert images to latents (cast to fp16 to match VAE dtype)
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float16)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
            
            # Sample noise and add to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise with autocast for fp16
            with autocast(device_type="cuda"):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()
                
                # Log step loss and learning rate
                step_loss = accumulated_loss
                step_losses.append(step_loss)
                learning_rates.append(scheduler.get_last_lr()[0])
                accumulated_loss = 0
                global_step += 1
                
                progress_bar.set_postfix({
                    'loss': f"{step_loss:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'step': global_step
                })
            
            total_train_loss += loss.item() * gradient_accumulation_steps
            
        except RuntimeError as e:
            print(f"Training error: {e}")
            raise e
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    epoch_train_losses.append(avg_train_loss)
    
    # Clear GPU memory after training phase
    torch.cuda.empty_cache()
    
    # --- VALIDATION PHASE ---
    unet.eval()
    total_val_loss = 0
    val_steps = 0
    
    with torch.no_grad(), autocast(device_type="cuda"):
        for batch in tqdm(val_dataloader, desc="Validation"):
            latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float16)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
            
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            val_loss = F.mse_loss(noise_pred.float(), noise.float())
            total_val_loss += val_loss.item()
            val_steps += 1
            
    avg_val_loss = total_val_loss / val_steps
    epoch_val_losses.append(avg_val_loss)
    
    current_lr = scheduler.get_last_lr()[0]
    print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

# --- 6. Save Model ---
unet.save_pretrained(args.lora_weights_out)
print(f"Training complete. Improved LoRA weights saved to {args.lora_weights_out}.")

# --- 7. Enhanced Plotting ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Epoch losses
ax1.plot(range(1, num_epochs + 1), epoch_train_losses, label='Training Loss', marker='o', linewidth=2)
ax1.plot(range(1, num_epochs + 1), epoch_val_losses, label='Validation Loss', marker='o', linewidth=2)
ax1.set_title('Training and Validation Loss over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Step-wise training loss
if len(step_losses) > 0:
    ax2.plot(step_losses, linewidth=1, alpha=0.7)
    ax2.set_title('Step-wise Training Loss')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('MSE Loss')
    ax2.grid(True, alpha=0.3)

# Learning rate schedule
if len(learning_rates) > 0:
    ax3.plot(learning_rates, linewidth=2, color='orange')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

# Loss trend (smoothed)
if len(step_losses) > 100:
    window_size = max(1, len(step_losses) // 50)
    smoothed = np.convolve(step_losses, np.ones(window_size)/window_size, mode='same')
    ax4.plot(smoothed, linewidth=2, color='green')
    ax4.set_title('Smoothed Training Loss Trend')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('MSE Loss')
    ax4.grid(True, alpha=0.3)
else:
    # Fallback: show loss distribution if we have enough data
    if len(step_losses) > 50:
        ax4.hist(step_losses[25:], bins=20, alpha=0.7, edgecolor='black')
        ax4.set_title('Training Loss Distribution')
        ax4.set_xlabel('MSE Loss')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(args.lora_train_results_out, dpi=300, bbox_inches='tight')
print(f"Enhanced analysis plot saved as '{args.lora_train_results_out}'.")

# Save training metrics
training_metrics = {
    'epoch_train_losses': epoch_train_losses,
    'epoch_val_losses': epoch_val_losses,
    'final_metrics': {
        'final_train_loss': epoch_train_losses[-1] if epoch_train_losses else None,
        'final_val_loss': epoch_val_losses[-1] if epoch_val_losses else None,
        'min_train_loss': min(epoch_train_losses) if epoch_train_losses else None,
        'min_val_loss': min(epoch_val_losses) if epoch_val_losses else None,
    },
    'hyperparameters': {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'effective_batch_size': batch_size * gradient_accumulation_steps,
        'lora_rank': lora_config.r,
        'lora_alpha': lora_config.lora_alpha,
        'lora_dropout': lora_config.lora_dropout,
        'num_epochs': num_epochs,
        'warmup_steps': warmup_steps,
        'total_steps': total_steps,
        'weight_decay': args.weight_decay,
        'eps': args.eps
    }
}

with open('training_results.json', 'w') as f:
    json.dump(training_metrics, f, indent=2)
print("Training metrics saved to 'training_results.json'.")

# Summary statistics
if epoch_train_losses and epoch_val_losses:
    print(f"\n--- Training Summary ---")
    print(f"Initial train loss: {epoch_train_losses[0]:.4f}")
    print(f"Final train loss: {epoch_train_losses[-1]:.4f}")
    print(f"Best train loss: {min(epoch_train_losses):.4f}")
    print(f"Initial val loss: {epoch_val_losses[0]:.4f}")
    print(f"Final val loss: {epoch_val_losses[-1]:.4f}")
    print(f"Best val loss: {min(epoch_val_losses):.4f}")
    
    # Check for improvement
    improvement = epoch_train_losses[0] - epoch_train_losses[-1]
    improvement_pct = (improvement / epoch_train_losses[0]) * 100
    print(f"Training loss improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
    
    if improvement > 0.01:  # At least 1% improvement
        print("✓ Significant training improvement detected!")
    else:
        print("⚠ Limited training improvement - consider adjusting hyperparameters")