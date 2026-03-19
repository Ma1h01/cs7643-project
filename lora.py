import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Import your custom dataset
from dataset import EldenRingDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

# --- 1. Load Standard SD Components ---
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Freeze base models
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# --- 2. Setup PEFT (LoRA Baseline) ---
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# --- 3. Setup DataLoaders ---
# Assuming your data is split into train.jsonl and val.jsonl
train_dataset = EldenRingDataset("combined_data", "train.jsonl", tokenizer)
val_dataset = EldenRingDataset("combined_data", "val.jsonl", tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
num_epochs = 10

# Tracking lists for plotting
epoch_train_losses = []
epoch_val_losses = []

# --- 4. Training & Validation Loop ---
for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    
    # --- TRAINING PHASE ---
    unet.train()
    total_train_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Convert images to latents
        latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        # Sample noise and add to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
        
        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Calculate MSE Loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    avg_train_loss = total_train_loss / len(train_dataloader)
    epoch_train_losses.append(avg_train_loss)
    
    # --- VALIDATION PHASE ---
    unet.eval()
    total_val_loss = 0
    
    # Disable gradient calculation for validation to save memory and compute
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
            
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            val_loss = F.mse_loss(noise_pred, noise)
            total_val_loss += val_loss.item()
            
    avg_val_loss = total_val_loss / len(val_dataloader)
    epoch_val_losses.append(avg_val_loss)
    
    print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

# --- 5. Save Model and Plot ---
unet.save_pretrained("elden_ring_lora_weights")
print("Training complete! Weights saved.")

# Plot and save the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), epoch_val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.savefig('lora_loss_curve.png')
print("Loss curve saved as 'lora_loss_curve.png'.")