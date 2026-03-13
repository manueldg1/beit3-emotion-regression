import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from transformers import XLMRobertaTokenizer
from timm.models import create_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal__
from timm.models.registry import register_model

# Dependencies: pip install torchscale timm transformers
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig

# ---------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------
# Path to the BEiT-3 SentencePiece model file
TOKENIZER_PATH = "beit3.spm" 
# Name of the checkpoint file you have in your folder
MODEL_NAME_OR_PATH = "beit3_large_indomain_patch16_224.pth"
# Output file in Word2Vec format for VecMap
OUTPUT_TXT = "beit3.txt"

# ---------------------------------------------------------------------
# Utility / config builders
# ---------------------------------------------------------------------
def trunc_normal_(tensor, mean: float = 0.0, std: float = 1.0) -> None:
    __call_trunc_normal__(tensor, mean=mean, std=std, a=-std, b=std)

def _get_large_config(img_size: int = 224,
                      patch_size: int = 16,
                      drop_path_rate: float = 0,
                      checkpoint_activations=None,
                      mlp_ratio: int = 4,
                      vocab_size: int = 64010,
                      **kwargs) -> EncoderConfig:
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
# Model wrapper
# ---------------------------------------------------------------------
class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.beit3.text_embed

@register_model
def beit3_large_indomain_patch16_224(pretrained: bool = False, **kwargs) -> BEiT3Wrapper:
    # We use the standard 224 resolution as per your filename
    args = _get_large_config(img_size=224, **kwargs) 
    model = BEiT3Wrapper(args, **kwargs)
    return model

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        return ckpt["model"]
    if "module" in ckpt:
        return ckpt["module"]
    return ckpt

def load_tokenizer(path: str) -> XLMRobertaTokenizer:
    print(f"Loading tokenizer. Using XLM-R large vocabulary mapping...")
    # BEiT-3 is compatible with XLM-R tokenizer
    return XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

def extract_sorted_vocab(tokenizer: XLMRobertaTokenizer) -> Tuple[List[str], List[int]]:
    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    tokens, ids = zip(*vocab_items)
    return list(tokens), list(ids)

def export_embeddings_txt(tokens: List[str], embeddings: torch.Tensor, out_path: str) -> None:
    embs = embeddings.detach().cpu().numpy()
    V, D = embs.shape
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(f"{V} {D}\n")
        for tok, vec in zip(tokens, embs):
            # Clean tokens to avoid issues with VecMap (Word2Vec format)
            clean_tok = tok.replace(" ", " ").replace("\n", "")
            vec_str = " ".join(f"{x:.6f}" for x in vec)
            fout.write(f"{clean_tok} {vec_str}\n")
    print(f"Successfully wrote {out_path} with {V} tokens and {D} dimensions")

def main():
    print("--- Step 1: Loading Tokenizer ---")
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    print("--- Step 2: Building BEiT-3 Model Architecture ---")
    model = create_model("beit3_large_indomain_patch16_224", vocab_size=64010)

    print(f"--- Step 3: Loading Weights from {MODEL_NAME_OR_PATH} ---")
    sd = load_checkpoint(MODEL_NAME_OR_PATH)

    # Check for position embedding size mismatch and interpolate if necessary
    if "beit3.encoder.embed_positions.A.weight" in sd:
        ckpt_pos_embed = sd["beit3.encoder.embed_positions.A.weight"]
        model_pos_embed = model.beit3.encoder.embed_positions.A.weight
        
        if ckpt_pos_embed.shape != model_pos_embed.shape:
            print(f"Position embedding mismatch detected. Interpolating: {ckpt_pos_embed.shape} -> {model_pos_embed.shape}")
            # Separate special tokens (3) from patch tokens
            extra_tokens = ckpt_pos_embed[:3]
            pos_tokens = ckpt_pos_embed[3:]
            old_grid = int(math.sqrt(pos_tokens.shape[0]))
            new_grid = int(math.sqrt(model_pos_embed.shape[0] - 3))
            
            pos_tokens = pos_tokens.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(new_grid, new_grid), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
            sd["beit3.encoder.embed_positions.A.weight"] = torch.cat((extra_tokens, pos_tokens), dim=0)

    model.load_state_dict(sd, strict=False)
    print("Weights loaded successfully.")
    
    print("--- Step 4: Exporting Text Embeddings ---")
    tokens, _ = extract_sorted_vocab(tokenizer)
    # Extract the 1024-dimensional text embedding weights
    embs_weight = model.get_input_embeddings().weight
    
    export_embeddings_txt(tokens, embs_weight, OUTPUT_TXT)

if __name__ == "__main__":
    main()
