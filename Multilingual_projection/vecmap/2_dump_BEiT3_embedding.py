"""
Export BEiT-3 text embeddings to Word2Vec text format ("beit3.txt"),
loading them from the VQA fine-tuned checkpoint
(beit3_large_patch16_480_vqa.pth).

IMPORTANT NOTES compared to the base version (patch16_224):
- The VQA checkpoint uses image resolution 480 (not 224), so the model
  config must have img_size=480.
- The text branch (beit3.text_embed) structure is identical to the
  pretraining model: VQA fine-tuning only adds a pooler and a
  classification head, which are not needed here and are ignored
  (strict=False in load_state_dict).
"""

import math
from typing import Tuple, List

import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer
from timm.models import create_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal__
from timm.models.registry import register_model

from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# 1) Path to the BEiT-3 SentencePiece model
TOKENIZER_DIR = "beit3.spm"
# 2) Path to the VQA checkpoint (patch16, 480px)
MODEL_NAME_OR_PATH = "beit3_large_patch16_480_vqa.pth"
# 3) Output file
OUTPUT_TXT = "beit3.txt"


# ---------------------------------------------------------------------
# Utility / config builders
# ---------------------------------------------------------------------
def trunc_normal_(tensor, mean: float = 0.0, std: float = 1.0) -> None:
    __call_trunc_normal__(tensor, mean=mean, std=std, a=-std, b=std)


def _get_large_config_480(img_size: int = 480,
                          patch_size: int = 16,
                          drop_path_rate: float = 0,
                          checkpoint_activations=None,
                          mlp_ratio: int = 4,
                          vocab_size: int = 64010,
                          **kwargs) -> EncoderConfig:
    """Same 'large' config as pretraining, but with img_size=480
    (resolution used by VQA fine-tuned checkpoints)."""
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


# ---------------------------------------------------------------------
# Model wrapper and registration
# ---------------------------------------------------------------------
class BEiT3Wrapper(nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self) -> None:

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self) -> int:
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight',
            'beit3.vision_embed.cls_token', 'logit_scale'
        }

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.beit3.text_embed


@register_model
def beit3_large_patch16_480(pretrained: bool = False, **kwargs) -> BEiT3Wrapper:
    """'Large' variant at 480 resolution, used by VQA checkpoints."""
    args = _get_large_config_480(**kwargs)
    args.normalize_output = False
    model = BEiT3Wrapper(args, **kwargs)
    return model


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def load_checkpoint(path: str):
    """Load a checkpoint and return its state dict."""
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        return ckpt["model"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def build_registered_model() -> nn.Module:
    model = create_model(
        "beit3_large_patch16_480",
        pretrained=False,
        drop_path_rate=0.1,
        vocab_size=64010,
    )
    return model


# ---------------------------------------------------------------------
# Tokenizer / vocab and export helpers
# ---------------------------------------------------------------------
def load_tokenizer(tokenizer_dir: str) -> XLMRobertaTokenizer:
    """Instantiate the XLM-R tokenizer from BEiT-3's .spm file."""
    return XLMRobertaTokenizer(tokenizer_dir)


def extract_sorted_vocab(
        tokenizer: XLMRobertaTokenizer) -> Tuple[List[str], List[int]]:
    """Return tokens and ids sorted by id.

    IMPORTANT: do NOT build this from tokenizer.get_vocab() — that
    returns a dict keyed by token STRING, so if two different ids
    happen to render to the same string, one of them is silently
    dropped and the resulting list is shorter than the true vocab
    size. Iterating by id instead guarantees one entry per id.
    """
    vocab_size = len(tokenizer)
    ids = list(range(vocab_size))
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in ids]
    return tokens, ids


def export_embeddings_txt(tokens: List[str], embeddings: torch.Tensor,
                          out_path: str) -> None:
    """Write Word2Vec-style text file.

    IMPORTANT: the checkpoint's embedding matrix may have more rows
    than the tokenizer's actual vocab size (e.g. padded to a multiple
    of 8 for hardware efficiency — extra rows with no corresponding
    token). We only ever write one line per real token, so the header
    count must reflect len(tokens), NOT embeddings.shape[0], or
    downstream readers (gensim/VecMap) will expect more vectors than
    are actually present in the file and fail.
    """
    embs = embeddings.detach().cpu().numpy()
    V_matrix, D = embs.shape
    N = len(tokens)

    if N > V_matrix:
        raise ValueError(
            f"More tokens ({N}) than embedding rows ({V_matrix}); "
            "cannot export.")
    if N < V_matrix:
        print(f"Note: embedding matrix has {V_matrix} rows but tokenizer "
              f"has {N} tokens ({V_matrix - N} unused/padding rows will "
              f"be skipped).")

    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(f"{N} {D}\n")
        for tok, vec in zip(tokens, embs):
            vec_str = " ".join(f"{x:.6f}" for x in vec)
            fout.write(f"{tok} {vec_str}\n")
    print(f"Wrote {out_path} with {N} tokens x {D} dimensions")


def main():
    print("Loading tokenizer")
    tokenizer = load_tokenizer(TOKENIZER_DIR)

    # Create the timm-registered model (large config, img_size=480)
    model = build_registered_model()
    print("Loading VQA checkpoint from", MODEL_NAME_OR_PATH)
    sd = load_checkpoint(MODEL_NAME_OR_PATH)

    # strict=False: the VQA checkpoint also contains a pooler/classification
    # head that are not part of the plain BEiT3Wrapper backbone.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Missing keys: {len(missing)} | Unexpected keys (e.g. VQA pooler/head): {len(unexpected)}")
    print("Model loaded")

    # Extract vocab sorted by id
    tokens, _ = extract_sorted_vocab(tokenizer)

    embs_weight = model.get_input_embeddings().weight

    # Export in Word2Vec text format
    export_embeddings_txt(tokens, embs_weight, OUTPUT_TXT)


if __name__ == "__main__":
    main()
