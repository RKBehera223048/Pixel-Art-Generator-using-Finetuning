import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

# Define paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "with_finetuning"
MODEL_PATH = BASE_DIR / "outputs" / "merged_model"

def main():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "generated_image_gpu_finetuned_test_new10.png"

    print(f"Loading Finetuned Stable Diffusion model from {MODEL_PATH} onto GPU...")
    
    # Initialize the Stable Diffusion pipeline using the locally merged finetuned model
    # Using torch.float16 reduces VRAM usage by 50% without losing quality!
    pipe = StableDiffusionPipeline.from_pretrained(
        str(MODEL_PATH), 
        torch_dtype=torch.float16,
        safety_checker=None, # Disable the safety checker to save memory and time
        requires_safety_checker=False
    )
    
    # Move the model to the GPU
    pipe = pipe.to("cuda")

    # Example prompt
    # Note: We automatically include the trigger words for your finetuned pixel art style
    # base_prompt = "A boy sitting on a bridge"
    # prompt = f"pixel_art, {base_prompt}, pixelated, 16-bit"
    # negative_prompt = "ugly, blurry, photorealistic, 3d render, smooth, modern, photograph"
    prompt = "a pixel art of attention_to_detail, depicting an old wizard reading in a cozy candlelit stone library, featuring tall bookshelves, a roaring fireplace, an owl stained glass window, and a sleeping golden dragon on a side table, 16-bit"
    negative_prompt = "ugly, blurry, photorealistic, 3d render"

    print(f"\nGenerating image on GPU...")
    print(f"Prompt: {prompt}")
    
    # Generate the image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=9.0,
        width=512,
        height=512
    ).images[0]

    # Save the output
    image.save(str(output_file))
    print(f"\nSuccess! Image saved to: {output_file}")

if __name__ == "__main__":
    main()
