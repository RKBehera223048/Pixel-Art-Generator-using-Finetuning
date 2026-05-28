# Pixel Art Generator: LoRA Fine-Tuning Pipeline

## Project Overview
This project establishes an end-to-end machine learning pipeline to fine-tune a Stable Diffusion v1.5 model to generate **Pixel Art** natively on a consumer-grade laptop GPU (NVIDIA RTX 4050). The pipeline bridges the entire lifecycle from raw dataset ingestion and captioning to low-VRAM model training and local inference testing.

---

## 1. Dataset Preparation (`prepare_dataset.py`)
The project started with a massive raw dataset (`kansallisgalleria-SCRT-Dataset-HF`) containing deeply nested directories of extremely high-resolution art pieces. The unique aspect of this dataset was that the image filenames themselves contained highly descriptive captions.

**What the script accomplishes:**
*   **Recursive Crawling:** It deeply scans through all subdirectories to find target images.
*   **Path-Length Handling:** Because Windows has strict path-length limits and the original filenames were enormous, the script safely renames the processed outputs sequentially (e.g., `image_00001.png`). 
*   **Caption Extraction:** It extracts the massive original filenames, prepends a global trigger word (`"pixel_art"`), and maps them to the sequential image IDs inside a Hugging Face-compatible `metadata.jsonl` file.
*   **Standardization:** It resizes the multi-megapixel images down to a standard 512x512 resolution (using Lanczos resampling) while bypassing PIL's decompression bomb safety limits to accommodate the huge source images.

## 2. Low-VRAM LoRA Fine-Tuning (`train_lora.py`)
Training a Diffusion model from scratch or performing full fine-tuning is impossible on a 6GB VRAM RTX 4050. To solve this, a highly optimized LoRA (Low-Rank Adaptation) training script was engineered.

**What the script accomplishes:**
*   **Targeted Weight Injection:** Uses Hugging Face `peft` to inject trainable LoRA matrices into the UNet's cross-attention layers and feed-forward networks (e.g., `to_q`, `to_k`, `to_v`, `ff.net.0.proj`). You recently expanded the rank (r=16) to capture finer stylistic details.
*   **Extreme Memory Optimization:** Freezes the Text Encoder and VAE, enables **Gradient Checkpointing** on the UNet, and replaces the standard optimizer with the `bitsandbytes` **8-bit AdamW** optimizer. It also utilizes PyTorch Automatic Mixed Precision (AMP) `autocast` in FP16. This combination makes local training possible.
*   **Automated Merging:** After running the specified number of steps (e.g., 3000 steps), the script automatically unloads and merges the learned LoRA weights back into the base pipeline, saving a standalone, ready-to-use model in `outputs/merged_model`.

## 3. Inference Testing (`generate_image_gpu*.py`)
To validate the success of the fine-tuning without relying on external web UIs, standalone inference scripts were built. 

**What the scripts accomplish:**
*   **Baseline vs. Finetuned:** The `generate_image_gpu.py` tests standard SD v1.5, while `generate_image_gpu_finetuned.py` tests the newly merged local model.
*   **Inference Optimization:** Both load the pipelines dynamically into VRAM using `torch.float16` and can enable attention slicing for faster, memory-safe generation.
*   **Prompt Automation:** The finetuned script allows for easy testing by taking base prompts (like *"beautiful selkie, ocean, mermaid"*) and injecting the necessary learned trigger modifiers (*"pixel_art, [prompt], 16-bit"*).

## Current Status
*   The architecture is 100% complete and verified. 
*   The dataset is fully prepared with generated metadata.
*   The training script has been successfully stress-tested on your GPU and is currently actively running the full 3000+ step training process!
