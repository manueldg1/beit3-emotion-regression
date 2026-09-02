"""
Build the seed dictionary of tokens shared between the BEiT-3 vocabulary
and the XLM-RoBERTa-Large vocabulary, used by VecMap to learn the
orthogonal transformation.
"""

BEIT3_TXT = "beit3.txt"
XLMR_TXT = "xlmr.txt"
OUT_SEED_PATH = "out_seed.txt"


def load_vocab(path):
    """
    Load the vocab (first column) from a word2vec-format file.
    Assumes the first line is "V D" header; skips it.
    """
    vocab = []
    with open(path, encoding='utf-8') as f:
        header = f.readline()  # e.g. "64010 1024"
        for line in f:
            tok = line.split(' ', 1)[0]
            vocab.append(tok)
    return vocab


def main(beit3_path, xlmr_path, out_seed_path):
    beit3_vocab = load_vocab(beit3_path)
    xlmr_vocab = load_vocab(xlmr_path)

    set_beit = set(beit3_vocab)
    set_xlmr = set(xlmr_vocab)

    # Intersection: common tokens
    common = sorted(set_beit & set_xlmr)

    print(f"BEiT-3 vocab size: {len(beit3_vocab)}")
    print(f"XLM-R vocab size:  {len(xlmr_vocab)}")
    print(f"Common tokens:     {len(common)}")

    # Write seed dictionary
    with open(out_seed_path, 'w', encoding='utf-8') as fout:
        for tok in common:
            fout.write(f"{tok} {tok}\n")

    print(f"Wrote seed dictionary to {out_seed_path}")


if __name__ == "__main__":
    main(BEIT3_TXT, XLMR_TXT, OUT_SEED_PATH)
