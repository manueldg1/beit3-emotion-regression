from transformers import AutoTokenizer, AutoModel
import numpy as np
from transformers import XLMRobertaTokenizer
import os
from timm.models import create_model

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from timm.models.registry import register_model
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig

XLRM_TOKENIZER_VOCAB_SIZE = 250002
XLMR_EMB_PATH = "xlmr_in_beit3_space.pt"


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
    """Overwrite model’s text_embed.weight with new_emb."""
    layer = model.beit3.text_embed
    assert layer.weight.shape == new_emb.shape, (
        f"Expected {layer.weight.shape}, got {new_emb.shape}")
    layer.weight.data.copy_(new_emb)


def load_beit3_model(model_name,
                     tokenizer_model_name="xlm-roberta-large",
                     xlmr_embeddings_path=XLMR_EMB_PATH,
                     checkpoint_path=None):
    """
    Load the BEiT3 model with the specified model name and checkpoint path.
    """
    # Load the tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_model_name)

    # Create the model
    model = create_model(
        model_name,
        drop_path_rate=0.1,
    )

    # Load the weights from the disk (source: 224px model)
    if checkpoint_path:
        sd = load_checkpoint_dict(checkpoint_path)

# ─── POSITIONAL EMBEDDING INTERPOLATION (224 -> 480) ───────────────────
        if "beit3.vision_embed.pos_embed" in sd:
            pos_embed_checkpoint = sd["beit3.vision_embed.pos_embed"]
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = 1 # CLS token (not part of the spatial grid)
            
            # Original grid size: sqrt(197 - 1) = 14
            old_grid_size = int(math.sqrt(pos_embed_checkpoint.shape[1] - num_extra_tokens))
            # Target grid size for 480px: sqrt(900) = 30
            new_grid_size = int(math.sqrt(model.beit3.vision_embed.num_patches))
            
            if old_grid_size != new_grid_size:
                print(f"Resizing pos_embed: {old_grid_size}x{old_grid_size} -> {new_grid_size}x{new_grid_size}")
                
                # Separate CLS token from spatial tokens
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                
                # Reshape to [Batch, Channels, Height, Width] for interpolation
                # Channels here is the embedding dimension (1024)
                pos_tokens = pos_tokens.reshape(-1, old_grid_size, old_grid_size, embedding_size).permute(0, 3, 1, 2)
                
                # Use bicubic interpolation to upscale the positional grid
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
                
                # Flatten back to sequence format: [Batch, New_Num_Patches, 1024]
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                
                # Concatenate back the CLS token and update the state dict
                sd["beit3.vision_embed.pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)

        # Remove pre-trained text embeddings to inject our aligned XLM-R vectors later
        sd.pop("beit3.text_embed.weight", None)
        
        # Load the weights into the model. 
        # strict=False allows missing weights (like the new regression head)
        model.load_state_dict(sd, strict=False)
        print("✓ Backbone loaded and interpolated for higher resolution.")

        # ─── INJECT MAPPED EMBEDDINGS ──────────────────────────────────────────
        # Load the XLM-R vectors already aligned to BEiT-3 space via VecMap
        new_emb = torch.load(xlmr_embeddings_path)
        inject_new_embeddings(model, new_emb)
        print("✓ Injected cross-lingual aligned embeddings.")

    return model, tokenizer

# ─── MAIN EXECUTION ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use the custom model registered for 480px resolution and regression tasks
    model_name = "beit3_large_patch16_480_valence_arousal" 
    tokenizer_model_name = "xlm-roberta-large"
    xlmr_embeddings_path = XLMR_EMB_PATH
    checkpoint_path = "/cfs/home/u036743/emotion_recognition/beit3_large_indomain_patch16_224.pth"

    # Initialize model and tokenizer
    model, tokenizer = load_beit3_model(model_name, tokenizer_model_name,
                                        xlmr_embeddings_path, checkpoint_path)

    print(f"Model {model_name} ready for fine-tuning at 480x480 resolution.")
