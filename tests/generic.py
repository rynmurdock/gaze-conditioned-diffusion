'''
python tests/generic.py
'''


import sys
import logging
import requests
import torch

import numpy as np
from PIL import Image
from diffusers import Flux2Transformer2DModel

sys.path.append('/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/src/')

# from utils.eval_utils import pil_to_n1_1_tensor, get_dinoscore


"""
Build a tiny, randomly-initialized stub of Flux2Transformer2DModel for testing.

Every config dimension is shrunk to 2-4 while preserving the internal
consistency constraints of the real architecture, so the model is a fully
valid, runnable (forward-passable) Flux.2 transformer -- just ~13k params
instead of billions.

Constraints that must hold (found by inspecting
diffusers/models/transformers/transformer_flux2.py):
  - sum(axes_dims_rope) must equal attention_head_dim
    (per-axis RoPE embeddings are concatenated, then applied across the
    full head dim)
  - each entry in axes_dims_rope must be even (rotary splits real/imag)
  - timestep_guidance_channels must be even (sinusoidal Timesteps embedding)
  - len(axes_dims_rope) is the number of position-id axes expected by
    img_ids / txt_ids at forward time (kept at 4, same as the real model)
"""

import torch
from diffusers import Flux2Transformer2DModel

TINY_CONFIG = dict(
    patch_size=1,
    in_channels=16,
    out_channels=None,            # resolves to in_channels
    num_layers=1,                 # double-stream blocks (real default: 8)
    num_single_layers=1,          # single-stream blocks (real default: 48)
    attention_head_dim=8,         # real default: 128
    num_attention_heads=2,        # real default: 48
    joint_attention_dim=4,        # real default: 15360
    timestep_guidance_channels=4, # real default: 256
    mlp_ratio=2.0,                # real default: 3.0
    axes_dims_rope=(2, 2, 2, 2),  # must sum to attention_head_dim
    rope_theta=2000,
    eps=1e-6,
    guidance_embeds=False,
)

def build_stub_transformer(seed: int = 0) -> Flux2Transformer2DModel:
    torch.manual_seed(seed)
    model = Flux2Transformer2DModel(**TINY_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stub Flux2Transformer2DModel built: {n_params:,} params")
    return model

def test_batched_rope():
    transformer = build_stub_transformer().to('cpu', torch.bfloat16)

    torch.manual_seed(7)
    latents = torch.randn((1, 16, 16,), device='cpu', dtype=torch.bfloat16)
    timesteps = torch.randint(0, 1000, (1,)).to(latents.device, latents.dtype)
    p_embs = torch.randn((1, 4, 4)).to(latents.device, latents.dtype)
    txt_ids = torch.randint(0, 100, (4, 4)).to(latents.device, latents.dtype)
    img_ids = torch.randint(0, 100, (16, 4)).to(latents.device, latents.dtype)

    batch_size_1_out = transformer(
        hidden_states=latents,  # (B, image_seq_len, C)
        timestep=timesteps / 1000,
        guidance=None,
        encoder_hidden_states=p_embs,
        txt_ids=txt_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]

    from modeling.klein_batched_rope import batchify_transformer_rope
    transformer = batchify_transformer_rope(transformer)

    two_latents = torch.randn((2, 16, 16,), device='cpu', dtype=torch.bfloat16)
    two_timesteps = torch.randint(0, 1000, (2,)).to(latents.device, latents.dtype)
    two_p_embs = torch.randn((2, 4, 4)).to(latents.device, latents.dtype)
    two_txt_ids = torch.randint(0, 100, (2, 4, 4)).to(latents.device, latents.dtype)
    two_img_ids = torch.randint(0, 100, (2, 16, 4)).to(latents.device, latents.dtype)

    latents = torch.cat([latents, two_latents])
    timesteps = torch.cat([timesteps, two_timesteps])
    p_embs = torch.cat([p_embs, two_p_embs])
    txt_ids = torch.cat([txt_ids[None], two_txt_ids])
    img_ids = torch.cat([img_ids[None], two_img_ids])

    batch_size_3_out = transformer(
            hidden_states=latents,  # (B, image_seq_len, C)
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states=p_embs,
            txt_ids=txt_ids,
            img_ids=img_ids,  # B, image_seq_len, 4
            return_dict=False,
        )[0]

    assert torch.equal(batch_size_1_out, batch_size_3_out[:1])

def test_dinoscore():
    # Get images from Figure 11
    urls = [
        'https://github.com/google/dreambooth/blob/main/dataset/rc_car/03.jpg?raw=true', # reference from Fig 11
        'https://github.com/google/dreambooth/blob/main/dataset/rc_car/02.jpg?raw=true'# Real Sample from Fig 11
    ]
    images = [Image.open(requests.get(url, stream=True).raw)for url in urls]
    metric = get_dinoscore(images)
    logging.info(f'''
    DINO Score
    Expected: 0.770
    Calculated: {metric:.3f}''')
    assert abs(metric - 0.770) < 0.001, (
                    f'Metric is {abs(metric - 0.770)} away from known good')

# on cpu rn; uncomment all other lines after
# test_dinoscore()
test_batched_rope()
