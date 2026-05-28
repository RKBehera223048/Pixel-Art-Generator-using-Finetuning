import os
import torch
from pathlib import Path
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading

# Define paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "pixel_art_xl"

# Model identifier on Hugging Face
MODEL_ID = "nerijs/pixel-art-xl"

# Base SDXL model (pixel-art-xl is a fine-tuned LoRA on top of SDXL)
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"Loading base model: {BASE_MODEL_ID} ...")

    # Load the base SDXL pipeline
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )

    # Use DPM++ 2M Karras scheduler for faster, high-quality sampling
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )

    print(f"Loading LoRA weights from: {MODEL_ID} ...")
    pipe.load_lora_weights(MODEL_ID)

    pipe = pipe.to(device)

    # Enable memory optimizations
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory-efficient attention.")
        except Exception:
            print("xformers not available, using default attention.")

    # Prompt — use "pixel art" trigger words for best results
    prompt = "pixel art, a cozy medieval tavern interior with a roaring fireplace, wooden tables and chairs, pixel art style"
    negative_prompt = "blurry, 3d render, photo, realistic, smooth, modern, ugly, deformed"

    num_inference_steps = 25
    guidance_scale = 7.5
    width = 512
    height = 512
    seed = 42

    print(f"\nGenerating pixel art image ...")
    print(f"  Prompt: {prompt}")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Seed: {seed}")

    # Set seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)

    # Generate the image
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    image = result.images[0]

    # Save the output
    output_file = OUTPUT_DIR / "pixel_art_xl_generated2.png"
    image.save(str(output_file))
    print(f"\nSuccess! Image saved to: {output_file}")

    # Also save generation metadata
    metadata_file = OUTPUT_DIR / "generation_metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Base Model: {BASE_MODEL_ID}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Negative Prompt: {negative_prompt}\n")
        f.write(f"Steps: {num_inference_steps}\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dtype: {dtype}\n")
    print(f"Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    main()
