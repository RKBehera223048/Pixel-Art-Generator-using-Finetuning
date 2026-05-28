import os
import glob
import json
import csv
from PIL import Image
from tqdm import tqdm

# Disable decompression bomb limits for huge images
Image.MAX_IMAGE_PIXELS = None

def main():
    input_dir = os.path.join("processed_data", "pixel art")
    output_dir = "processed_data_resized"
    
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images (assuming png, jpg, jpeg)
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                
    print(f"Found {len(image_paths)} images in '{input_dir}'.")
    
    # Load metadata mapping from the previous AI captioning step
    metadata_csv_path = os.path.join(input_dir, "_metadata.csv")
    generated_captions = {}
    if os.path.exists(metadata_csv_path):
        with open(metadata_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                generated_captions[row['new_filename']] = row['caption']
                
    metadata = []
    
    for idx, path in enumerate(tqdm(image_paths, desc="Processing images")):
        try:
            filename = os.path.basename(path)
            
            # If we generated a rich caption, use it. Otherwise fallback.
            if filename in generated_captions:
                # Still prepend the 'pixel_art' style trigger for Stable Diffusion LoRA
                caption = f"pixel_art, {generated_captions[filename]}"
            else:
                name_without_ext = os.path.splitext(filename)[0]
                clean_name = name_without_ext.replace('_', ' ')
                caption = f"pixel_art, {clean_name}"
            
            img = Image.open(path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Resize to 512x512
            img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Save to output_dir with sequential name to avoid OS path length issues
            out_filename = f"image_{idx:05d}.png"
            out_path = os.path.join(output_dir, out_filename)
            img_resized.save(out_path)
            
            # Add to metadata
            metadata.append({
                "file_name": out_filename,
                "text": caption
            })
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    # Write HuggingFace metadata.jsonl
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saved {len(metadata)} resized images and metadata.jsonl to '{output_dir}'.")

if __name__ == "__main__":
    main()
