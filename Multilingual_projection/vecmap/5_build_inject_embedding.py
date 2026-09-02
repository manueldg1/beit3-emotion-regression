"""
05_build_and_inject_embeddings.py

Final step:
1) Build the new embedding matrix (XLM-R mapped into BEiT-3's space)
   from VecMap's output.
2) Load the VQA fine-tuned BEiT-3 checkpoint
   (beit3_large_patch16_480_vqa.pth, 480 resolution) and inject the new
   matrix in place of the original text embeddings.

NOTES for the VQA checkpoint:
- img_size=480 (not 224) in the model config.
- The checkpoint also contains a VQA pooler/classification head, which
  is unrelated to the text branch and is ignored via strict=False.
- The text branch key is still "beit3.text_embed.weight", identical to
  the pretraining model, so the injection logic is unchanged.
"""

import math
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from transformers import XLMRobertaTokenizer
from timm.models import create_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal__
from timm.models.registry import register_model

from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig

# --- Config ----------------------------------------------------------------
XLMR_MAPPED_TXT = "xlmr_mapped.txt"
BEIT3_TXT = "beit3.txt"
XLMR_TOKENIZER_DIR = "xlm-roberta-large"
NEW_EMB_OUT_FILE = "xlmr_in_beit3_space.pt"

XLMR_TOKENIZER_VOCAB_SIZE = 250002
CHECKPOINT_PATH = "beit3_large_patch16_480_vqa.pth"
MODEL_NAME = "beit3_large_patch16_480"

# Output: full model checkpoint (backbone + VQA head/pooler + new XLM-R
# embeddings), ready to be reloaded directly for fine-tuning.
FINAL_MODEL_OUT = "beit3_large_patch16_480_vqa_xlmr_projected.pth"


# --- Part 1: build the final embedding matrix ------------------------------
def build_mapped_embedding_matrix() -> torch.Tensor:
    """
    Build the final 250002 x 1024 embedding matrix for the XLM-R
    vocabulary, with a critical distinction:

    - SHARED tokens (tokens that already existed, with the exact same
      spelling, in BEiT-3's original 64010-token vocabulary): copy
      BEiT-3's ORIGINAL embedding for that token EXACTLY, unchanged.
      These tokens already have a perfect representation in BEiT-3's
      space -- there is nothing to project, and doing so would only
      throw away information (this was the bug flagged by the
      professor: shared tokens must keep bit-identical embeddings,
      cosine similarity == 1.0 by construction, not by approximation).

    - NEW tokens (present in XLM-R's vocabulary but NOT in BEiT-3's
      original vocabulary): use the orthogonal-Procrustes-mapped
      vector (xlmr_mapped.txt), which was fit using the shared tokens
      as anchor pairs. This is the only part of the pipeline that is
      an approximation, and the only part we cannot directly verify
      against a known ground truth (BEiT-3 never had these tokens).
    """
    print("Loading mapped XLM-R vectors from", XLMR_MAPPED_TXT)
    mapped_kv = KeyedVectors.load_word2vec_format(XLMR_MAPPED_TXT, binary=False)

    print("Loading ORIGINAL BEiT-3 embeddings (for shared tokens) from", BEIT3_TXT)
    beit3_lookup = {}
    with open(BEIT3_TXT, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split(" ")
            beit3_lookup[parts[0]] = [float(x) for x in parts[1:]]

    print("Loading XLM-RoBERTa-large tokenizer from", XLMR_TOKENIZER_DIR)
    tok = XLMRobertaTokenizer.from_pretrained(XLMR_TOKENIZER_DIR)

    vocab = tok.get_vocab()  # { token_str: token_id }
    vocab_size = len(vocab)
    D = mapped_kv.vector_size  # should be 1024

    assert D == 1024, f"Expected mapped vectors to be 1024-dim, got {D}"

    print(f"Building final embedding matrix: {vocab_size} tokens x {D} dims")
    new_emb = torch.zeros(vocab_size, D, dtype=torch.float32)

    n_shared_copied = 0
    n_new_projected = 0
    missing = 0
    for token, idx in vocab.items():
        if token in beit3_lookup:
            # SHARED token: exact copy of BEiT-3's original embedding.
            new_emb[idx] = torch.tensor(beit3_lookup[token], dtype=torch.float32)
            n_shared_copied += 1
        elif token in mapped_kv:
            # NEW token: use the Procrustes-projected XLM-R embedding.
            new_emb[idx] = torch.from_numpy(mapped_kv[token].copy())
            n_new_projected += 1
        else:
            # stray token found in neither source: random init
            new_emb[idx].normal_(0, 0.02)
            missing += 1

    print(f"  -> {n_shared_copied} shared tokens: copied BEiT-3's original embedding exactly")
    print(f"  -> {n_new_projected} new tokens: used the Procrustes-projected embedding")
    print(f"  -> {missing} tokens found in neither source; randomly initialized")

    print("Saving new embeddings to", NEW_EMB_OUT_FILE)
    torch.save(new_emb, NEW_EMB_OUT_FILE)
    return new_emb


# --- Part 2: BEiT-3 model definition (large config, img_size=480) ----------
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal__(tensor, mean=mean, std=std, a=-std, b=std)


def _get_large_config_480(img_size=480,
                          patch_size=16,
                          drop_path_rate=0,
                          checkpoint_activations=None,
                          mlp_ratio=4,
                          vocab_size=XLMR_TOKENIZER_VOCAB_SIZE,
                          **kwargs):
    return EncoderConfig(
        img_size=img_size,
        patch_size=patch_size,
        vocab_size=vocab_size,
        multiway=True,
        layernorm_embedding=False,
        normalize_output=True,
        no_output_layer=True,
        drop_path_rate=drop_path_rate,
        encoder_embed_dim=1024,
        encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio),
        encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )


class BEiT3Wrapper(nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight',
            'beit3.vision_embed.cls_token', 'logit_scale'
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_input_embeddings(self):
        return self.beit3.text_embed


@register_model
def beit3_large_patch16_480(pretrained=False, **kwargs):
    """'Large' variant at 480 resolution, used by VQA checkpoints."""
    args = _get_large_config_480(**kwargs)
    args.normalize_output = False
    model = BEiT3Wrapper(args, **kwargs)
    return model


def load_checkpoint_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        return ckpt["model"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def inject_new_embeddings(model: nn.Module, new_emb: torch.Tensor):
    """Overwrite model's text_embed.weight with new_emb."""
    layer = model.get_input_embeddings()
    assert layer.weight.shape == new_emb.shape, (
        f"Expected {layer.weight.shape}, got {new_emb.shape}")
    layer.weight.data.copy_(new_emb)


# --- Part 3: load the VQA checkpoint + inject embeddings -------------------
def load_beit3_vqa_model(model_name=MODEL_NAME,
                         tokenizer_model_name="xlm-roberta-large",
                         xlmr_embeddings_path=NEW_EMB_OUT_FILE,
                         checkpoint_path=CHECKPOINT_PATH):
    """
    Load the BEiT-3 model (VQA variant, 480px) and inject the XLM-R
    embeddings mapped into BEiT-3's space.
    """
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_model_name)

    model = create_model(
        model_name,
        drop_path_rate=0.1,
        vocab_size=XLMR_TOKENIZER_VOCAB_SIZE,
    )

    if checkpoint_path:
        sd = load_checkpoint_dict(checkpoint_path)

        # Remove text_embed.weight from the state dict: it will be replaced
        sd.pop("beit3.text_embed.weight", None)

        # strict=False: the VQA checkpoint also includes a pooler/
        # classification head not present in the plain BEiT3Wrapper
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Backbone loaded (excluding text_embed). "
              f"Missing: {len(missing)}, unexpected (e.g. VQA pooler/head): {len(unexpected)}")

        # Inject mapped embeddings
        # Load new XLM-R -> BEiT-3 embeddings
        new_emb = torch.load(xlmr_embeddings_path)

        # Confirm vocab size matches XLM-R tokenizer
        assert len(tokenizer) == new_emb.shape[0], (
            f"Tokenizer vocab ({len(tokenizer)}) "
            f"!= embeddings ({new_emb.shape[0]})")

        inject_new_embeddings(model, new_emb)
        print("Injected new embeddings.")

    return model, tokenizer


def save_full_model(model: nn.Module, out_path: str) -> None:
    """Save the assembled model's full state dict to disk, so it can be
    reloaded later (e.g. for fine-tuning) without repeating the whole
    mapping/injection pipeline.

    Wrapped under a "model" key, matching the format used by the
    original BEiT-3 checkpoints (load_checkpoint_dict looks for this
    key first).
    """
    torch.save({"model": model.state_dict()}, out_path)
    print(f"Saved full model (backbone + VQA head/pooler + new XLM-R "
          f"embeddings) to {out_path}")


def main():
    # 1) build the mapped embedding matrix (if not already present)
    build_mapped_embedding_matrix()

    # 2) load the BEiT-3 VQA model (480px) and inject the embeddings
    model, tokenizer = load_beit3_vqa_model()

    print("Model and tokenizer loaded successfully (VQA 480px checkpoint).")

    # 3) persist the fully assembled model to disk for later use
    #    (e.g. fine-tuning in a separate script/session)
    save_full_model(model, FINAL_MODEL_OUT)


if __name__ == "__main__":
    main()
