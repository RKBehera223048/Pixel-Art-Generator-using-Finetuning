"""
Image Processor for Diffusion Model Training Data
===================================================
Uses the Gemini API with DUAL models (Gemma 4 26B + Gemma 4 31B) to analyze
art images and rename them with detailed descriptions suitable for training
a diffusion model.

Images are distributed round-robin across both models, each with its own
rate limiter, effectively doubling throughput.

Reads images from the raw_data folder and writes renamed copies
to the processed_data folder. Supports resuming from where it left off.
"""

import os
import re
import io
import json
import time
import shutil
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types
from PIL import Image

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data" / "kansallisgalleria-art-images-dataset" / "kansallisgalleria-SCRT-Dataset-HF"
OUTPUT_DIR = BASE_DIR / "processed_data" / "kansallisgalleria-SCRT-Dataset-HF"
PROGRESS_FILE = OUTPUT_DIR / "_processing_progress.json"
METADATA_CSV = OUTPUT_DIR / "_metadata.csv"

# Dual-model configuration — each model gets its own rate limit
MODELS = [
    {"id": "gemma-4-26b-a4b-it", "label": "Gemma4-26B", "rpm": 5, "concurrent": 1},
    {"id": "gemma-4-31b-it",     "label": "Gemma4-31B", "rpm": 5, "concurrent": 1},
]

MAX_FILENAME_LEN = 180       # max filename length (before extension)
MAX_RETRIES = 5              # retries per image on failure

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("image_processor")

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# ─── Prompt ──────────────────────────────────────────────────────────────────

CAPTION_PROMPT = """You are an expert art captioner creating training data for a diffusion model.

Analyze this artwork image and produce a SINGLE detailed caption that describes:
1. The subject matter (what is depicted - people, landscapes, objects, scenes)
2. The art style and medium (oil painting, watercolor, sketch, etc.)
3. The color palette and lighting (warm tones, cool blues, dramatic shadows, etc.)
4. The composition and mood (serene, dramatic, intimate, grand, etc.)
5. Any notable artistic techniques or features

Rules:
- Write ONE continuous sentence or two short sentences (no bullet points)
- Be specific and descriptive, avoid vague terms
- Use natural language suitable for text-to-image model training
- Keep it between 30-80 words
- Do NOT include the artist name or title
- Do NOT start with "This is" or "The image shows"
- Focus on visual content that would help a model recreate this image

Example output format:
"A serene Finnish lakeside landscape at golden hour, oil on canvas with loose impressionist brushwork, featuring birch trees reflecting in still water with warm amber and soft violet tones, creating a peaceful contemplative atmosphere"

Now describe this artwork:"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def sanitize_filename(text: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """Convert a caption into a safe filename."""
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', text)
    text = re.sub(r'[\s_]+', '_', text)
    text = text.strip('_').strip('.')
    if len(text) > max_len:
        text = text[:max_len].rsplit('_', 1)[0]
    return text


def load_progress() -> dict:
    """Load processing progress from disk."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed": {}, "failed": {}, "skipped": []}

def get_nvapi_processed_files() -> set:
    """Read the NVAPI progress file to see what it has already processed."""
    nvapi_file = OUTPUT_DIR / "_processing_progress_nvapi.json"
    if nvapi_file.exists():
        try:
            with open(nvapi_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get("processed", {}).keys())
        except Exception as e:
            log.warning(f"Could not read NVAPI progress file: {e}")
            return set()
    return set()


def save_progress(progress: dict):
    """Save processing progress to disk."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def save_metadata_csv(progress: dict):
    """Export a CSV mapping original filenames to captions."""
    with open(METADATA_CSV, 'w', encoding='utf-8') as f:
        f.write("original_filename,new_filename,caption\n")
        for orig, info in sorted(progress["processed"].items()):
            caption_escaped = info["caption"].replace('"', '""')
            new_name = info.get("new_filename", "")
            f.write(f'"{orig}","{new_name}","{caption_escaped}"\n')


def get_image_files() -> list[Path]:
    """Get all image files from the raw directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    files = []
    for f in sorted(RAW_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(f)
    return files


# ─── Rate Limiter ────────────────────────────────────────────────────────────

class RateLimiter:
    """Simple token-bucket rate limiter for API calls."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.last_call + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_call = time.monotonic()


# ─── Model Worker ────────────────────────────────────────────────────────────

class ModelWorker:
    """A worker that processes images using a specific model with its own rate limiter."""

    def __init__(self, client: genai.Client, model_id: str, label: str, rpm: int, concurrent: int):
        self.client = client
        self.model_id = model_id
        self.label = label
        self.rate_limiter = RateLimiter(rpm)
        self.semaphore = asyncio.Semaphore(concurrent)
        self.success_count = 0
        self.fail_count = 0

    async def caption_image(self, image_path: Path) -> str:
        """Send an image to this model and get a descriptive caption."""
        async with self.semaphore:
            await self.rate_limiter.acquire()

            # Read and prepare image → convert to JPEG bytes
            img = Image.open(image_path)
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")

            # Resize large images to reduce API payload (max 1024px on longest side)
            max_side = 1024
            if max(img.size) > max_side:
                ratio = max_side / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to JPEG bytes
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            image_bytes = buf.getvalue()

            # Call Gemini API
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_id,
                contents=[
                    CAPTION_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                ),
            )

            caption = response.text.strip() if response.text else ""
            return caption


# ─── Core Processing ────────────────────────────────────────────────────────

async def process_single_image(
    worker: ModelWorker,
    image_path: Path,
    progress: dict,
    progress_lock: asyncio.Lock,
    index: int,
    total: int,
    duplicate_counter: dict,
) -> bool:
    """Process a single image: caption → rename → copy."""
    filename = image_path.name

    # Skip if already processed
    if filename in progress["processed"]:
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            caption = await worker.caption_image(image_path)

            if not caption:
                log.warning(f"[{index}/{total}] Empty caption for {filename}, retrying...")
                continue

            # Build new filename
            safe_name = sanitize_filename(caption)
            if not safe_name:
                safe_name = f"artwork_{index}"

            ext = image_path.suffix.lower()

            # Handle duplicate filenames (thread-safe)
            async with progress_lock:
                final_name = f"{safe_name}{ext}"
                if final_name in duplicate_counter:
                    duplicate_counter[final_name] += 1
                    final_name = f"{safe_name}_{duplicate_counter[final_name]}{ext}"
                else:
                    duplicate_counter[final_name] = 0

            # Copy file with new name
            dest = OUTPUT_DIR / final_name
            shutil.copy2(image_path, dest)

            # Record progress (thread-safe)
            async with progress_lock:
                progress["processed"][filename] = {
                    "caption": caption,
                    "new_filename": final_name,
                    "model": worker.model_id,
                }

            worker.success_count += 1
            log.info(f"[{index}/{total}] ✓ [{worker.label}] {filename} → {final_name[:70]}...")
            return True

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES:
                wait_time = 2 ** attempt  # exponential backoff
                log.warning(
                    f"[{index}/{total}] [{worker.label}] Attempt {attempt}/{MAX_RETRIES} "
                    f"failed for {filename}: {error_msg[:100]}. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                log.error(f"[{index}/{total}] ✗ [{worker.label}] FAILED {filename}: {error_msg[:150]}")
                async with progress_lock:
                    progress["failed"][filename] = {
                        "error": error_msg[:500],
                        "attempts": attempt,
                        "model": worker.model_id,
                    }
                worker.fail_count += 1
                return False


async def main():
    """Main processing pipeline with dual-model workers."""
    # Load environment with override to ensure it picks up new keys
    load_dotenv(BASE_DIR / ".env", override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not found in .env file!")
        return
    
    masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 15 else "***"
    log.info(f"Using API Key: {masked_key}")

    # Initialize client
    client = genai.Client(api_key=api_key)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load progress
    progress = load_progress()
    progress_lock = asyncio.Lock()

    # Get image files
    image_files = get_image_files()
    total = len(image_files)
    already_done = len(progress["processed"])

    # Initialize model workers
    workers = []
    for cfg in MODELS:
        w = ModelWorker(client, cfg["id"], cfg["label"], cfg["rpm"], cfg["concurrent"])
        workers.append(w)

    total_rpm = sum(cfg["rpm"] for cfg in MODELS)
    total_concurrent = sum(cfg["concurrent"] for cfg in MODELS)

    log.info(f"{'='*60}")
    log.info(f"  Image Processor — DUAL MODEL MODE")
    log.info(f"{'='*60}")
    log.info(f"  Source:     {RAW_DIR}")
    log.info(f"  Output:     {OUTPUT_DIR}")
    for i, cfg in enumerate(MODELS):
        log.info(f"  Model {i+1}:    {cfg['id']} ({cfg['rpm']} RPM, {cfg['concurrent']} workers)")
    log.info(f"  Combined:   {total_rpm} RPM, {total_concurrent} workers")
    log.info(f"  Total:      {total} images")
    log.info(f"  Done:       {already_done} (resuming)")
    log.info(f"  Remaining:  {total - already_done}")
    log.info(f"{'='*60}")

    if already_done >= total:
        log.info("All images already processed! Nothing to do.")
        save_metadata_csv(progress)
        return

    duplicate_counter: dict[str, int] = {}

    # Pre-populate duplicate counter with already-processed names
    for info in progress["processed"].values():
        name = info.get("new_filename", "")
        if name:
            duplicate_counter[name] = duplicate_counter.get(name, 0)

    # Filter to only unprocessed images
    remaining = [f for f in image_files if f.name not in progress["processed"]]
    log.info(f"Starting processing of {len(remaining)} remaining images...")

    # Process in batches for periodic progress saving
    SAVE_EVERY = 25
    start_time = time.monotonic()
    num_workers = len(workers)

    batch_size = SAVE_EVERY
    for batch_start in range(0, len(remaining), batch_size):
        # Dynamically check what NVAPI has finished so we don't collide
        nvapi_processed_dynamic = get_nvapi_processed_files()

        batch = remaining[batch_start:batch_start + batch_size]
        tasks = []
        for i, img_path in enumerate(batch):
            if img_path.name in nvapi_processed_dynamic:
                log.info(f"Skipping {img_path.name} (Already processed by NVAPI)")
                continue

            # global_idx represents how many are done now, just for logging
            global_idx = already_done + len(nvapi_processed_dynamic) + batch_start + i + 1
            # Round-robin: alternate images between workers
            worker = workers[i % num_workers]
            task = process_single_image(
                worker, img_path, progress, progress_lock,
                global_idx, total, duplicate_counter,
            )
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)


        # Save progress periodically
        save_progress(progress)

        # Log batch summary
        elapsed = time.monotonic() - start_time
        total_done = len(progress["processed"])
        rate = total_done / (elapsed / 60) if elapsed > 0 else 0
        remaining_count = total - total_done
        eta_mins = remaining_count / rate if rate > 0 else 0

        worker_stats = " | ".join(
            f"{w.label}: {w.success_count}✓ {w.fail_count}✗"
            for w in workers
        )
        log.info(
            f"  ── Batch done │ {total_done}/{total} │ "
            f"{rate:.1f} img/min │ ETA: {eta_mins:.0f} min │ {worker_stats}"
        )

    # Final save
    save_progress(progress)
    save_metadata_csv(progress)

    # Summary
    elapsed = time.monotonic() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"  PROCESSING COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"  Processed:  {len(progress['processed'])}")
    log.info(f"  Failed:     {len(progress['failed'])}")
    log.info(f"  Time:       {elapsed/60:.1f} minutes")
    for w in workers:
        log.info(f"  {w.label}:  {w.success_count} success, {w.fail_count} failed")
    log.info(f"  Metadata:   {METADATA_CSV}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
