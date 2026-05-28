import os
import torch
from diffusers import StableDiffusionPipeline
import argparse

def generate_image(prompt, model_path="outputs/merged_model", out_path="outputs/samples/out.png", steps=20, cfg=7.5, seed=42):
    print(f"Loading merged model from {model_path}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the merged pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    # Optional memory optimizations for inference
    if device == "cuda":
        pipe.enable_attention_slicing()

    full_prompt = f"pixel_art, {prompt}, pixelated, 16-bit"
    neg_prompt = "blurry, photorealistic, 3d render, smooth"
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"Generating image for prompt: '{full_prompt}'")
    
    image = pipe(
        prompt=full_prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator
    ).images[0]
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path)
    print(f"Saved generated image to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the mountains", help="The prompt to generate")
    parser.add_argument("--out", type=str, default="outputs/samples/out.png", help="Output path")
    parser.add_argument("--model", type=str, default="outputs/merged_model", help="Path to the merged model")
    
    args = parser.parse_args()
    generate_image(args.prompt, model_path=args.model, out_path=args.out)
