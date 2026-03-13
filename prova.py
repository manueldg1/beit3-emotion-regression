import json
import os

# Configuration
jsonl_file = 'beit3_final_train.jsonl'
# The base directory where your images are actually stored
base_image_dir = os.path.join(os.getcwd(), "wikiart")

count_exists = 0
count_missing = 0
total_artemis = 0
total_textual = 0

print(f"Analyzing {jsonl_file}...")

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        raw_path = item.get('image_path')
        
        # Skip textual-only samples (samples with no image_path)
        if raw_path is None:
            total_textual += 1
            continue
            
        total_artemis += 1
        
        # --- PATH CORRECTION LOGIC (Same as in datasets.py) ---
        # 1. Normalize slashes: convert Windows backslashes to Unix forward slashes
        clean_path = raw_path.replace('\\', '/')
        parts = clean_path.split('/')
        
        # 2. Extract the relative path 'Style/Image_Name.jpg'
        # We take the last two components of the original path
        if len(parts) >= 2:
            relative_file_path = os.path.join(parts[-2], parts[-1])
            
            # 3. Construct the final absolute path on the current server
            actual_path = os.path.join(base_image_dir, relative_file_path)
            
            if os.path.exists(actual_path):
                count_exists += 1
            else:
                # Print the first 3 errors for debugging purposes
                if count_missing < 3:
                    print(f"DEBUG - Not found: {actual_path}")
                count_missing += 1
        else:
            count_missing += 1

# Statistics Output
print(f"\n--- FINAL RESULTS ---")
print(f"Text-only samples (skipped): {total_textual}")
print(f"Samples with image reference: {total_artemis}")
print(f"Images FOUND successfully: {count_exists} ({(count_exists/total_artemis)*100:.2f}%)")
print(f"Images MISSING: {count_missing} ({(count_missing/total_artemis)*100:.2f}%)")

if count_exists > 0:
    print("\n SUCCESS! The model can now 'see' the images. You can restart training.")
else:
    print("\n ERROR: Still 0%. Check if 'wikiart' contains subfolders like 'Cubism'.")
