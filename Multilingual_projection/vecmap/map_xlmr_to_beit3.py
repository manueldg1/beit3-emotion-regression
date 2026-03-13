'''
python3 map_embeddings.py --emnlp2016 \
  out_seed.txt \
  beit3.txt \
  xlmr.txt \
  beit3_mapped.txt \
  xlmr_mapped.txt
'''


import os
import torch
from gensim.models import KeyedVectors
from transformers import XLMRobertaTokenizer

# ─── Config ────────────────────────────────────────────────────────────────────
XLMR_MAPPED_TXT = "xlmr_mapped.txt"
XLMR_TOKENIZER_DIR = "xlm-roberta-large"
OUT_FILE = "xlmr_in_beit3_space.pt"

# ─── 1) load mapped vectors ────────────────────────────────────────────────────
print("Loading mapped XLM-R vectors from", XLMR_MAPPED_TXT)
mapped_kv = KeyedVectors.load_word2vec_format(XLMR_MAPPED_TXT, binary=False)

# ─── 2) load XLM-RoBERTa tokenizer ───────────────────────────────────────────────
print("Loading XLM-RoBERTa-large tokenizer from", XLMR_TOKENIZER_DIR)
tok = XLMRobertaTokenizer.from_pretrained(XLMR_TOKENIZER_DIR)

vocab = tok.get_vocab()  # { token_str: token_id }
vocab_size = len(vocab)
D = mapped_kv.vector_size  # should be 1024

assert D == 1024, f"Expected mapped vectors to be 1024-dim, got {D}"

# ─── 3) allocate + fill ───────────────────────────────────────────────────────
print(f"Building new embedding matrix: {vocab_size} tokens × {D} dims")
new_emb = torch.zeros(vocab_size, D, dtype=torch.float32)

missing = 0
for token, idx in vocab.items():
    if token in mapped_kv:
        new_emb[idx] = torch.from_numpy(mapped_kv[token])
    else:
        # random init for any stray tokens
        new_emb[idx].normal_(0, 0.02)
        missing += 1

print(
    f"  → {missing} tokens were not found in the mapped file; randomly initialized them"
)

# ─── 4) save ───────────────────────────────────────────────────────────────────
print("Saving new embeddings to", OUT_FILE)
torch.save(new_emb, OUT_FILE)
print("Done.")
