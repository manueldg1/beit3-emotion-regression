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

    def get_input_embeddings(self):
        return self.beit3.text_embed

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
def beit3_large_patch16_224(pretrained=False, **kwargs):
    args = _get_large_config(**kwargs)
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
    """Overwrite model’s text_embed.weight with new_emb."""
    layer = model.get_input_embeddings()
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

    # Load the checkpoint if provided
    if checkpoint_path:
        sd = load_checkpoint_dict(checkpoint_path)

        # --- Interpolation Logic for 480x480 resolution ---
        # The checkpoint usually has 199 positions (14x14 grid) from 224px training.
        # The model with 480px expects 903 positions (30x30 grid).
        pos_key = "beit3.encoder.embed_positions.A.weight"
        
        if pos_key in sd:
            ckpt_pos_embed = sd[pos_key]  # Source weights from checkpoint
            model_pos_embed = model.state_dict()[pos_key] # Target shape in the model
            
            if ckpt_pos_embed.shape[0] != model_pos_embed.shape[0]:
                print(f"INFO: Interpolating {pos_key} from {ckpt_pos_embed.shape[0]} to {model_pos_embed.shape[0]} positions.")
                
                # Calculate grid dimensions (assuming square grids)
                old_grid = int(math.sqrt(ckpt_pos_embed.shape[0])) # e.g., 14
                new_grid = int(math.sqrt(model_pos_embed.shape[0])) # e.g., 30
                dim = ckpt_pos_embed.shape[1]

                # Reshape to [Batch, Channels, Height, Width] for torch.nn.functional.interpolate
                # The embedding dimension (1024) is treated as channels
                pos_tokens = ckpt_pos_embed.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
                
                # Apply bicubic interpolation to resize the positional grid
                pos_tokens = F.interpolate(
                    pos_tokens, 
                    size=(new_grid, new_grid), 
                    mode='bicubic', 
                    align_corners=False
                )

                # Reshape back to [New_Sequence_Length, Embedding_Dim]
                sd[pos_key] = pos_tokens.permute(0, 2, 3, 1).reshape(-1, dim)
        
        # Remove the text_embed.weight from the state dict
        # Because we will inject the new embeddings
        sd.pop("beit3.text_embed.weight", None)

        model.load_state_dict(sd, strict=False)
        print("✓ Backbone loaded (excluding text_embed and with interpolated positions).")

        # Inject mapped embeddings
        new_emb = torch.load(xlmr_embeddings_path)

        # Confirm vocab size matches XLM-R tokenizer
        assert len(tokenizer) == new_emb.shape[0], (
            f"Tokenizer vocab ({len(tokenizer)}) "
            f"!= embeddings ({new_emb.shape[0]})")

        # Inject new embeddings
        inject_new_embeddings(model, new_emb)
        print("✓ Injected new embeddings.")

    return model, tokenizer        


if __name__ == "__main__":
    # Example usage
    model_name = "beit3_large_patch16_224"
    tokenizer_model_name = "xlm-roberta-large"
    xlmr_embeddings_path = XLMR_EMB_PATH
    checkpoint_path = "/cfs/home/u036743/unilm/beit3/beit3_large_indomain_patch16_224.pth"

    model, tokenizer = load_beit3_model(model_name, tokenizer_model_name,
                                        xlmr_embeddings_path, checkpoint_path)

    print("Model and tokenizer loaded successfully.")
