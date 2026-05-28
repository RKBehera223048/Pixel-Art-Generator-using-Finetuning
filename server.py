"""
Pixel Art LoRA Generator — Flask Backend
Serves the frontend and provides GPU inference via the finetuned Stable Diffusion model.
"""

import os
import io
import base64
import time
import glob
from pathlib import Path

from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

import torch
from diffusers import StableDiffusionPipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = BASE_DIR / "outputs" / "merged_model"
GALLERY_DIR = BASE_DIR / "output" / "with_finetuning"

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)

# ---------------------------------------------------------------------------
# Global Pipeline (loaded once)
# ---------------------------------------------------------------------------
_pipe = None


def get_pipeline():
    """Load the finetuned Stable Diffusion pipeline once and cache it globally."""
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"[server] Loading finetuned model from {MODEL_PATH} ...")
    _pipe = StableDiffusionPipeline.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    _pipe = _pipe.to("cuda")
    print("[server] Model loaded and ready on GPU.")
    return _pipe


# ---------------------------------------------------------------------------
# Routes — Static Frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(str(FRONTEND_DIR), path)


# ---------------------------------------------------------------------------
# Routes — Gallery
# ---------------------------------------------------------------------------
@app.route("/gallery/<path:filename>")
def gallery_image(filename):
    """Serve a generated image from the gallery directory."""
    return send_from_directory(str(GALLERY_DIR), filename)


@app.route("/api/gallery")
def gallery_list():
    """Return a JSON list of all gallery image filenames (newest first)."""
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(
        GALLERY_DIR.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return jsonify([img.name for img in images])


# ---------------------------------------------------------------------------
# Routes — Image Generation
# ---------------------------------------------------------------------------
@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate a pixel art image from a prompt using the finetuned model."""
    data = request.get_json(force=True)

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    negative_prompt = data.get("negative_prompt", "ugly, blurry, photorealistic, 3d render")
    steps = int(data.get("steps", 50))
    guidance_scale = float(data.get("guidance_scale", 9.0))
    width = int(data.get("width", 512))
    height = int(data.get("height", 512))

    # Clamp values to safe ranges
    steps = max(10, min(steps, 150))
    guidance_scale = max(1.0, min(guidance_scale, 20.0))
    width = max(256, min(width, 1024))
    height = max(256, min(height, 1024))
    # Round dimensions to nearest 64
    width = (width // 64) * 64
    height = (height // 64) * 64

    try:
        pipe = get_pipeline()
        start = time.time()

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )
        image = result.images[0]
        elapsed = round(time.time() - start, 2)

        # Auto-save to gallery
        GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        save_name = f"generated_{timestamp}.png"
        image.save(str(GALLERY_DIR / save_name))

        # Encode as base64 for the frontend
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "image": b64,
            "filename": save_name,
            "elapsed": elapsed,
            "params": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
            },
        })

    except Exception as e:
        print(f"[server] Generation error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  PIXEL_LORA Generator Server")
    print(f"  Frontend: {FRONTEND_DIR}")
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Gallery:  {GALLERY_DIR}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
