import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
import streamlit as st

# Define paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "merged_model"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output" / "with_finetuning"

# 1. Page Configuration
st.set_page_config(
    page_title="Wizard Diffusion Studio",
    page_icon="🧙‍♂️",
    layout="wide"
)

# 2. Cached Model Loader
@st.cache_resource(show_spinner="Loading Finetuned Stable Diffusion model onto GPU...")
def load_pipeline(model_path):
    """Loads the diffusion model once and caches it in memory."""
    if not Path(model_path).exists():
        st.error(f"Model path not found: {model_path}. Please check your directory structure.")
        st.stop()
        
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_path), 
        torch_dtype=torch.float16,
        safety_checker=None, 
        requires_safety_checker=False
    )
    pipe = pipe.to("cuda")
    return pipe

# --- UI Layout ---

st.title("🧙‍♂️ Pixel Art Generation Studio")
st.caption("Generate custom 16-bit artwork using your locally fine-tuned Stable Diffusion model.")

# 3. Sidebar for parameters Configuration
st.sidebar.header("⚙️ Generation Parameters")

# Image Dimensions
width = st.sidebar.slider("Width", min_value=256, max_value=1024, value=512, step=64)
height = st.sidebar.slider("Height", min_value=256, max_value=1024, value=512, step=64)

# Inference Controls
num_inference_steps = st.sidebar.slider("Inference Steps", min_value=10, max_value=150, value=50, step=5)
guidance_scale = st.sidebar.slider("Guidance Scale (CFG)", min_value=1.0, max_value=20.0, value=9.0, step=0.5)

# Output Management
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Save Settings")
save_output = st.sidebar.checkbox("Save copy to local folder", value=True)
filename_input = st.sidebar.text_input("Filename", "generated_pixel_art.png")

# 4. Main Area: Prompts & Execution
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prompt Engineering")
    
    prompt = st.text_area(
        "Positive Prompt", 
        value="a pixel art of attention_to_detail,_depicting_a_solitary_knight_in_full_plate_armor_with_a_red_cape,_standing_on_a_mountain_peak_and_holding_a_greatsword,_at_sunset_over_a_sea_of_clouds_with_stars_appearing, 16-bit",
        height=200
    )
    
    negative_prompt = st.text_area(
        "Negative Prompt", 
        value="ugly, blurry, photorealistic, 3d render,digital art",
        height=80
    )
    
    generate_btn = st.button("✨ Generate Artwork", type="primary", use_container_width=True)

with col2:
    st.subheader("Output Preview")
    # Empty placeholder to cleanly hold the image or loading state
    preview_placeholder = st.empty()
    preview_placeholder.info("Adjust your prompts and click 'Generate Artwork' to begin.")

# 5. Generation Logic execution
if generate_btn:
    try:
        # Load pipeline (uses cache if already loaded)
        pipe = load_pipeline(MODEL_PATH)
        
        with st.spinner("🧙‍♂️ The Wizard is painting your request..."):
            # Run inference
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            )
            image = result.images[0]
            
            # Display image in the preview column
            with col2:
                preview_placeholder.image(image, caption="Generated Image", use_container_width=True)
                st.success("Generation complete!")
            
            # Handle local file saving if enabled
            if save_output:
                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                full_save_path = DEFAULT_OUTPUT_DIR / filename_input
                image.save(str(full_save_path))
                st.info(f"💾 Copy securely saved locally to: `{full_save_path}`")
                
    except Exception as e:
        st.error(f"An error occurred during generation: {str(e)}")