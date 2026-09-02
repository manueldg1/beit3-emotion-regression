"""
Run VecMap (Artetxe et al. 2016) to learn the orthogonal transformation
that aligns the BEiT-3 and XLM-RoBERTa-Large embeddings, using the seed
dictionary generated in the previous step.

Requires the VecMap repository cloned locally:
    git clone https://github.com/artetxem/vecmap.git

Usage:
    python 4_run_vecmap.py --vecmap-dir /path/to/vecmap
"""


import numpy as np

SEED_DICT_PATH = "out_seed.txt"
BEIT3_TXT = "beit3.txt"
XLMR_TXT = "xlmr.txt"
XLMR_MAPPED_OUT = "xlmr_mapped.txt"


def load_word2vec(path: str):
    """Load a word2vec-format txt file.
    Returns (tokens: List[str], vectors: np.ndarray of shape (V, D))."""
    tokens = []
    vecs = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()  # "V D"
        for line in f:
            parts = line.rstrip("\n").split(" ")
            tokens.append(parts[0])
            vecs.append([float(x) for x in parts[1:]])
    return tokens, np.array(vecs, dtype=np.float64)


def load_seed_tokens(path: str):
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if parts and parts[0]:
                tokens.append(parts[0])
    return tokens


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Classic orthogonal Procrustes solution.
    Returns R (D x D, orthogonal) minimizing || A @ R - B ||_F.
    """
    M = A.T @ B                       # (D, D)
    U, S, Vt = np.linalg.svd(M)       # M = U S V^T
    R = U @ Vt
    return R


def main():
    print("Loading BEiT-3 embeddings (target, fixed space) from", BEIT3_TXT)
    beit3_tokens, beit3_vecs = load_word2vec(BEIT3_TXT)
    beit3_idx = {tok: i for i, tok in enumerate(beit3_tokens)}

    print("Loading XLM-R embeddings (source, to be rotated) from", XLMR_TXT)
    xlmr_tokens, xlmr_vecs = load_word2vec(XLMR_TXT)
    xlmr_idx = {tok: i for i, tok in enumerate(xlmr_tokens)}

    print("Loading seed dictionary (shared tokens) from", SEED_DICT_PATH)
    seed_tokens = load_seed_tokens(SEED_DICT_PATH)

    # Build paired matrices A (XLM-R) and B (BEiT-3) for the seed tokens,
    # in matching order, using the RAW (non-normalized) vectors.
    A_rows, B_rows = [], []
    skipped = 0
    for tok in seed_tokens:
        if tok in xlmr_idx and tok in beit3_idx:
            A_rows.append(xlmr_vecs[xlmr_idx[tok]])
            B_rows.append(beit3_vecs[beit3_idx[tok]])
        else:
            skipped += 1

    A = np.stack(A_rows)  # (n_seed, D)
    B = np.stack(B_rows)  # (n_seed, D)
    print(f"Built {A.shape[0]} seed pairs (skipped {skipped}), dim={A.shape[1]}")

    print("Solving orthogonal Procrustes (BEiT-3 space kept fixed)...")
    R = orthogonal_procrustes(A, B)

    # Sanity check on the seed set itself
    residual = np.linalg.norm(A @ R - B) / np.linalg.norm(B)
    print(f"Relative residual on seed set: {residual:.4f} "
          f"(lower is better; near 0 = near-perfect alignment on seed tokens)")

    print("Applying R to the full XLM-R vocabulary "
          f"({xlmr_vecs.shape[0]} tokens)...")
    xlmr_mapped = xlmr_vecs @ R

    print("Writing", XLMR_MAPPED_OUT)
    V, D = xlmr_mapped.shape
    with open(XLMR_MAPPED_OUT, "w", encoding="utf-8") as fout:
        fout.write(f"{V} {D}\n")
        for tok, vec in zip(xlmr_tokens, xlmr_mapped):
            vec_str = " ".join(f"{x:.6f}" for x in vec)
            fout.write(f"{tok} {vec_str}\n")

    print(f"Done. Wrote {XLMR_MAPPED_OUT} with {V} tokens x {D} dimensions.")
    print("BEiT-3's own embeddings were NOT modified (used as-is, fixed target).")


if __name__ == "__main__":
    main()
