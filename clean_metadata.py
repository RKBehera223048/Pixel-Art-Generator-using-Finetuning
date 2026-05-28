import json
import re
import os

def clean_text(text):
    # Keywords to remove (case-insensitive)
    phrases_to_remove = [
        r'\bpixel_art\b',
        r'\bdigital\b',
        r'\bartwork\b',
        r'\bpixel art\b',
        r'\babstract\b',
        r'\bcartoon\b',
        r'\bcartoon-style\b',
        r'\biconic\b',
        r'\bvideo game\b',
        r'\bvideo game art\b',
        r'\bgame\b',
        r'\billustration\b',
        r'\billustrated\b',
        r'\bdigitally\b',
        r'\bpixelated\b',
        r'\bpixel\b',
        r'\bart\b',
        r'\b8-bit\b',
        r'\b8 bit\b',
        r'\blow resolution\b'
    ]
    
    clean = text
    for phrase in phrases_to_remove:
        clean = re.sub(phrase, '', clean, flags=re.IGNORECASE)
        
    # Clean up punctuation and spacing left behind
    clean = re.sub(r'\s+', ' ', clean)       # collapse multiple spaces
    clean = re.sub(r'\s+,', ',', clean)      # remove spaces before commas
    clean = re.sub(r',+', ',', clean)        # collapse multiple commas
    clean = re.sub(r'^[\s,]+', '', clean)    # remove leading commas or spaces
    
    return clean.strip()

def main():
    filepath = r"processed_data_resized\metadata.jsonl"
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    cleaned_lines = []
    for line in lines:
        data = json.loads(line)
        original = data['text']
        cleaned = clean_text(original)
        data['text'] = cleaned
        cleaned_lines.append(json.dumps(data) + '\n')
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
        
    print(f"Cleaned {len(cleaned_lines)} lines in {filepath}")
    
    # Print a few examples to verify
    print("\nExample cleaned texts:")
    for i in range(min(5, len(cleaned_lines))):
        print(json.loads(cleaned_lines[i])['text'])

if __name__ == "__main__":
    main()
