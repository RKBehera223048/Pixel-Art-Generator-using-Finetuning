import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import CLIPTextModel, CLIPTokenizer
import bitsandbytes as bnb
from tqdm import tqdm

class PixelArtDataset(Dataset):
    def __init__(self, data_dir, metadata_file, tokenizer, size=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.size = size
        self.data = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
                    
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, item['file_name'])
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.transform(image)
        
        text_ids = self.tokenizer(
            item['text'], padding="max_length", truncation=True, 
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.squeeze()
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_ids
        }

def main():
    base_model = "runwayml/stable-diffusion-v1-5"
    data_dir = "processed_data_resized"
    metadata_file = os.path.join(data_dir, "metadata.jsonl")
    
    # ── Model load ──
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_enc = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # Move to GPU and set eval mode where appropriate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    text_enc.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)
    text_enc.eval()
    vae.eval()
    
    # UNet to GPU in float32 for training (or float16 if fully using mixed precision, but peft handles some of this)
    unet.to(device, dtype=torch.float32)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_enc.requires_grad_(False)
    unet.requires_grad_(False)

    # ── Inject LoRA ──
    print("Injecting LoRA...")
    lora_cfg = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.05,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        # target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "proj_out"],
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    # ── Memory optimisations ──
    unet.enable_gradient_checkpointing()
    optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=1e-4)

    # ── Dataset & DataLoader ──
    print("Loading dataset...")
    dataset = PixelArtDataset(data_dir, metadata_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ── Training Loop ──
    epochs = 4
    max_train_steps = len(dataloader) * epochs
    checkpoint_every = 1000           # save every N steps
    print(f"Starting training for {epochs} epochs (to reach {max_train_steps} steps)...")
    
    global_step = 0
    unet.train()
    
    # Using mixed precision (autocast) for the forward pass
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            if global_step >= max_train_steps:
                break
                
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            input_ids = batch["input_ids"].to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # Encode images to latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                timesteps = timesteps.long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_enc(input_ids)[0]

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            
            if global_step % 100 == 0:
                print(f"Step {global_step}/{max_train_steps} | Loss: {loss.item():.4f}")
                
            if global_step % checkpoint_every == 0:
                ckpt_path = f"outputs/checkpoints/step_{global_step}"
                os.makedirs(ckpt_path, exist_ok=True)
                unet.save_pretrained(ckpt_path)
                print(f"Checkpoint saved at step {global_step}")

        if global_step >= max_train_steps:
            break

    # ── Save & Merge ──
    print("Training finished! Saving and merging LoRA weights...")
    lora_save_path = "outputs/checkpoints/final"
    os.makedirs(lora_save_path, exist_ok=True)
    unet.save_pretrained(lora_save_path)
    
    print("Loading base pipeline for merging...")
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32).to("cpu")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_save_path)
    pipe.unet = pipe.unet.merge_and_unload()
    
    merged_path = "outputs/merged_model"
    os.makedirs(merged_path, exist_ok=True)
    pipe.save_pretrained(merged_path)
    print(f"Merged model saved to {merged_path}")

if __name__ == "__main__":
    main()
