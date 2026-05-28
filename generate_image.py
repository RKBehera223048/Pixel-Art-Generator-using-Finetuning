import os
from pathlib import Path
from stable_diffusion_cpp import StableDiffusion

# Define paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stable-diffusion-v1-5-GGUF" / "stable-diffusion-v1-5-Q4_1.gguf"
OUTPUT_DIR = BASE_DIR / "output" / "without_finetuning"

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure the GGUF file is in the stable-diffusion-v1-5-GGUF folder.")
        return

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = OUTPUT_DIR / "generated_image_q4.png"

    print(f"Loading Stable Diffusion model from: {MODEL_PATH.name} ...")
    
    # Initialize the Stable Diffusion model
    # Note: If you have a GPU and compiled with cuBLAS, it will use it automatically.
    stable_diffusion = StableDiffusion(
        model_path=str(MODEL_PATH),
        n_threads=os.cpu_count() // 2  # Use half of available CPU threads
    )

    # Example prompt based on the art dataset we've been processing
    prompt = "A boy sitting on a bridge"
    negative_prompt = "ugly, blurry, low resolution, bad quality, modern, unreal"

    print(f"\nGenerating image...")
    print(f"Prompt: {prompt}")
    
    # Generate the image
    images = stable_diffusion.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=512,
        sample_steps=20,
        cfg_scale=7.0
    )

    # Save the output
    if images and len(images) > 0:
        images[0].save(str(output_file))
        print(f"\nSuccess! Image saved to: {output_file}")
    else:
        print("\nFailed to generate image.")

if __name__ == "__main__":
    main()
