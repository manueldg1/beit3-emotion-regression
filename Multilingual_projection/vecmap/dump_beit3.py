import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from transformers import XLMRobertaTokenizer
from timm.models import create_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig

# Add the BEiT-3 root directory to sys.path to allow importing modeling_finetune
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modeling_finetune import BEiT3ForValenceArousalRegression

XLRM_TOKENIZER_VOCAB_SIZE = 250002
XLMR_EMB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'xlmr_in_beit3_space.pt'))

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def _get_large_config(img_size=224,
                      patch_size=16,
                      drop_path_rate=0,
                      checkpoint_activations=None,
                      mlp_ratio=4,
                      vocab_size=XLRM_TOKENIZER_VOCAB_SIZE,
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

@register_model
def beit3_large_patch16_480_valence_arousal(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, vocab_size=250002, **kwargs)
    args.normalize_output = False
    model = BEiT3ForValenceArousalRegression(args, **kwargs)
    return model

def load_checkpoint_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        return ckpt["model"]
    if "state_dict" in ckpt:
                return ckpt["state_dict"]
    return ckpt

def inject_new_embeddings(model: nn.Module, new_emb: torch.Tensor):
    """
    Overwrite the text embedding weights within the BEiT-3 core.
    In your architecture, this layer is located at model.beit3.text_embed.
    """
    if hasattr(model, 'beit3') and hasattr(model.beit3, 'text_embed'):
        layer = model.beit3.text_embed
        print("✓ Successfully accessed model.beit3.text_embed.")
    else:
        # Fallback if structure differs
        raise AttributeError("Could not find 'text_embed' layer. Check BEiT3Wrapper structure.")

    assert layer.weight.shape == new_emb.shape, (
        f"Dimension mismatch! Expected {layer.weight.shape}, got {new_emb.shape}")

    # Copy the mapped XLM-R embeddings into the model
    layer.weight.data.copy_(new_emb)


def load_beit3_model(model_name,
                     tokenizer_model_name="xlm-roberta-large",
                     xlmr_embeddings_path=XLMR_EMB_PATH,
                     checkpoint_path=None):
    """
    Load the BEiT-3 model, clean the VQA checkpoint, and inject mapped embeddings.
    """
    # Load the tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_model_name)
    # Create the custom regression model
    # This uses your registered 'beit3_large_patch16_480_valence_arousal'
    model = create_model(
        model_name,
        drop_path_rate=0.1,
    )

    # Load the checkpoint if provided
    if checkpoint_path:
        state_dict = load_checkpoint_dict(checkpoint_path)

        # 1. Remove the original text embedding weight
        state_dict.pop("beit3.text_embed.weight", None)

        # 2. Filter out non-backbone parameters (VQA heads, etc.)
        # We only keep keys starting with 'beit3.' to protect your custom heads/pooler
        backbone_keys = [k for k in state_dict.keys() if k.startswith("beit3.")]
        clean_sd = {k: state_dict[k] for k in backbone_keys}

        print(f"✓ Extracted {len(clean_sd)} backbone parameters from checkpoint.")

        # Load the cleaned state dict into the model
        # strict=False allows your new MLP heads to remain initialized with your scale (0.001)
        model.load_state_dict(clean_sd, strict=False)
        print("✓ Backbone loaded successfully.")

        # 3. Inject the mapped XLM-R into BEiT-3 embeddings
        new_embeddings = torch.load(xlmr_embeddings_path)
                
        # Confirm vocab size matches
        assert len(tokenizer) == new_embeddings.shape[0], (
            f"Tokenizer vocab ({len(tokenizer)}) != embeddings ({new_embeddings.shape[0]})")

        inject_new_embeddings(model, new_embeddings)
        print("✓ Injected new XLM-R embeddings.")

    return model, tokenizer

if __name__ == "__main__":
    # Example usage
    model_name = "beit3_large_patch16_480_valence_arousal"
    tokenizer_model_name = "xlm-roberta-large"
    xlmr_embeddings_path = XLMR_EMB_PATH
    checkpoint_path = "/cfs/home/u036743/unilm/beit3/Multilingual_projection/vecmap/beit3_large_patch16_480_vqa.pth"

    model, tokenizer = load_beit3_model(model_name, tokenizer_model_name,
                                        xlmr_embeddings_path, checkpoint_path)

    print("Model and tokenizer loaded successfully.")
