# create_seed.py

def load_vocab(path):
    """Load the vocab (first column) from a word2vec-format file."""
    vocab = []
    with open(path, encoding='utf-8') as f:
        # Skip the header (e.g., "64010 1024")
        f.readline()
        for line in f:
            # Get the first word of the line
            tok = line.split(' ', 1)[0]
            vocab.append(tok)
    return vocab

def main():
    # Paths to your generated files
    beit3_path = "beit3.txt"
    xlmr_path = "xlmr.txt" # Assicurati che questo file esista già!
    out_seed_path = "out_seed.txt"

    print("Loading vocabularies...")
    beit3_vocab = load_vocab(beit3_path)
    xlmr_vocab = load_vocab(xlmr_path)

    # Find common tokens using sets for speed
    common = sorted(set(beit3_vocab) & set(xlmr_vocab))

    print(f"BEiT-3 vocab size: {len(beit3_vocab)}")
    print(f"XLM-R vocab size:  {len(xlmr_vocab)}")
    print(f"Common tokens found: {len(common)}")

    # Write the seed dictionary in 'word word' format
    with open(out_seed_path, 'w', encoding='utf-8') as fout:
        for tok in common:
            fout.write(f"{tok} {tok}\n")

    print(f"Done! Seed dictionary saved to {out_seed_path}")

if __name__ == "__main__":
    main()