import json
import random
import os

# Configuration
BASE_PATH = os.path.expanduser("~/emotion_dataset/processed_data")
OUTPUT_PATH = os.path.expanduser("~/emotion_dataset/processed_data")

# 1. PRE-SPLIT DATASETS (Respecting original distributions)
PRE_SPLIT_MAP = {
    "MELD_train.jsonl": "train", "MELD_dev.jsonl": "val", "MELD_test.jsonl": "test",
    "AffectNet_train_set.jsonl": "train", "AffectNet_validation_set.jsonl": "val"
}
PRE_SPLIT_NAMES = set(PRE_SPLIT_MAP.keys())

def load_and_standardize(filename):
    """Standardizes columns to: Text, Image_path, Valence, Arousal"""
    standardized = []
    path = os.path.join(BASE_PATH, filename)
    if not os.path.exists(path):
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Key mapping with case-insensitive fallback
                text = data.get("Text") or data.get("text") or ""
                val = data.get("Valence") or data.get("valence") or 0.0
                aro = data.get("Arousal") or data.get("arousal") or 0.0
                
                # Image Path cleaning
                raw_path = data.get("Image_path") or data.get("Image_Path")
                clean_path = raw_path.replace("\\/", "/") if raw_path else None
                
                standardized.append({
                    "Text": text,
                    "Image_path": clean_path,
                    "Valence": float(val),
                    "Arousal": float(aro)
                })
            except:
                continue
    return standardized

def main():
    final_train, final_val, final_test = [], [], []
    
    # Get all .jsonl files in processed_data
    all_files = [f for f in os.listdir(BASE_PATH) if f.endswith('.jsonl') and f not in ["train.jsonl", "val.jsonl", "test.jsonl"]]
    
    print(f"--- Dataset Unification Started ---")

    for filename in all_files:
        # Case A: Dataset is already split (MELD, AffectNet)
        if filename in PRE_SPLIT_NAMES:
            target = PRE_SPLIT_MAP[filename]
            data = load_and_standardize(filename)
            if target == "train": final_train.extend(data)
            elif target == "val": final_val.extend(data)
            elif target == "test": final_test.extend(data)
            print(f"[PRE-SPLIT] {filename:40} -> Added to {target}")
            
        # Case B: Monolithic dataset (Everything else)
        else:
            data = load_and_standardize(filename)
            if not data:
                continue
            
            # Local shuffle for this specific dataset before splitting
            random.seed(42)
            random.shuffle(data)
            
            n = len(data)
            train_end = int(n * 0.8)
            val_end = int(n * 0.9)
            
            final_train.extend(data[:train_end])
            final_val.extend(data[train_end:val_end])
            final_test.extend(data[val_end:])
            print(f"[SPLIT 80/10/10] {filename:36} -> T:{train_end}, V:{val_end-train_end}, Test:{n-val_end}")

    # Final global shuffle for each split to mix all sources
    random.shuffle(final_train)
    random.shuffle(final_val)
    random.shuffle(final_test)

    # Saving final files
    outputs = [("train.jsonl", final_train), ("val.jsonl", final_val), ("test.jsonl", final_test)]
    print("\n" + "="*50)
    for name, content in outputs:
        out_path = os.path.join(OUTPUT_PATH, name)
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in content:
                f.write(json.dumps(item) + '\n')
        print(f"SUCCESS: {name} created with {len(content)} samples.")

if __name__ == "__main__":
    main()
