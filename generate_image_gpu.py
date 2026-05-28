import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

# Define paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "without_finetuning"

def main():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "generated_image_gpu4_test_new0.png"

    print("Loading standard Stable Diffusion v1.5 model onto GPU...")
    print("If this is your first time, it will download the model files (~4GB) automatically.")
    
    # Initialize the Stable Diffusion pipeline using the standard safetensors version
    # Using torch.float16 reduces VRAM usage by 50% without losing quality!
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16,
        safety_checker=None, # Disable the safety checker to save memory and time
        requires_safety_checker=False
    )
    
    # Move the model to the GPU
    pipe = pipe.to("cuda")

    # Example prompt
    # prompt = "A beautiful serene Finnish lakeside landscape at golden hour, oil on canvas, loose impressionist brushwork, birch trees reflecting in still water, warm amber and soft violet tones"
    # negative_prompt = "ugly, blurry, low resolution, bad quality, modern, photograph"
    # base_prompt = "A boy sitting on a bridge"
    # prompt = f"pixel_art, {base_prompt}, pixelated, 16-bit"
    # negative_prompt = "ugly, blurry, photorealistic, 3d render, smooth, modern, photograph"
    prompt = "a pixel art of attention_to_detail,_depicting_a_solitary_knight_in_full_plate_armor_with_a_red_cape,_standing_on_a_mountain_peak_and_holding_a_greatsword,_at_sunset_over_a_sea_of_clouds_with_stars_appearing, 16-bit"
    negative_prompt = "ugly, blurry, photorealistic, 3d render,digital art"


    print(f"\nGenerating image on GPU...")
    print(f"Prompt: {prompt}")
    
    # Generate the image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        guidance_scale=7.0,
        width=512,
        height=512
    ).images[0]

    # Save the output
    image.save(str(output_file))
    print(f"\nSuccess! Image saved to: {output_file}")

if __name__ == "__main__":
    main()