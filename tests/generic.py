'''
python tests/generic.py
'''


import sys
import logging
import requests
import torch

from PIL import Image
from diffusers import Flux2Transformer2DModel

sys.path.append('/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/src/')



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

def test_attention_mask_and_batched_rope():
    transformer = build_stub_transformer().to('cpu', torch.bfloat16)
    
    torch.manual_seed(7)
    latents = torch.randn((1, 7, 16,), device='cpu', dtype=torch.bfloat16)
    timesteps = torch.randint(0, 1000, (1,)).to(latents.device, latents.dtype)
    p_embs = torch.randn((1, 4, 4)).to(latents.device, latents.dtype)
    txt_ids = torch.randint(0, 100, (4, 4)).to(latents.device, latents.dtype)
    img_ids = torch.randint(0, 100, (7, 4)).to(latents.device, latents.dtype)

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

    two_latents = torch.randn((1, 18, 16,), device='cpu', dtype=torch.bfloat16)
    two_timesteps = torch.randint(0, 1000, (1,)).to(latents.device, latents.dtype)
    two_p_embs = torch.randn((1, 4, 4)).to(latents.device, latents.dtype)
    two_txt_ids = torch.randint(0, 100, (1, 4, 4)).to(latents.device, latents.dtype)
    two_img_ids = torch.randint(0, 100, (1, 18, 4)).to(latents.device, latents.dtype)

    ones_ls = [torch.ones_like(l) for l in [latents[0], two_latents[0]]]
    latents_there_mask = torch.nn.utils.rnn.pad_sequence(ones_ls, 
                                                         batch_first=True).squeeze(1)

    latents = torch.nn.utils.rnn.pad_sequence([latents[0], two_latents[0]], 
                                                    batch_first=True, padding_value=10000).squeeze(1)

    latents_neg520_padded = torch.nn.utils.rnn.pad_sequence([latents[0], two_latents[0]], 
                                                    batch_first=True, padding_value=-520).squeeze(1)

    diff_latents = torch.nn.utils.rnn.pad_sequence([torch.randn_like(latents[0]), torch.randn_like(two_latents[0])], 
                                                    batch_first=True, padding_value=10000).squeeze(1)
    
    timesteps = torch.cat([timesteps, two_timesteps])
    p_embs = torch.cat([p_embs, two_p_embs])
    txt_ids = torch.cat([txt_ids[None], two_txt_ids])
    img_ids = torch.nn.utils.rnn.pad_sequence([img_ids, two_img_ids[0]], 
                                                    batch_first=True,).squeeze(1)

    attn_mask = latents_there_mask.sum(-1)
    attn_mask = torch.nn.functional.pad(attn_mask, (4, 0,), value=1) != 0.

    batch_size_3_out = transformer(
            hidden_states=latents,  # (B, image_seq_len, C)
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states=p_embs,
            txt_ids=txt_ids,
            img_ids=img_ids,  # B, image_seq_len, 4
            joint_attention_kwargs={'attention_mask':attn_mask},
            return_dict=False,
        )[0]

    batch_size_3_out_sans_mask = transformer(
                hidden_states=latents,  # (B, image_seq_len, C)
                timestep=timesteps / 1000,
                guidance=None,
                encoder_hidden_states=p_embs,
                txt_ids=txt_ids,
                img_ids=img_ids,  # B, image_seq_len, 4
                return_dict=False,
            )[0]
    
    batch_size_3_out_sans_mask = batch_size_3_out_sans_mask[:, : latents.size(1) :]
    batch_size_3_out = batch_size_3_out[:, : latents.size(1) :]
    batch_size_1_out = batch_size_1_out[:, : latents.size(1) :]
    assert torch.equal(batch_size_1_out, batch_size_3_out[:1, :batch_size_1_out.shape[1]]), (batch_size_1_out - batch_size_3_out[:1, :batch_size_1_out.shape[1]]).abs().max()

    # invariant to padding
    batch_size_3_out_diff_padding = transformer(
                        hidden_states=latents_neg520_padded,  # (B, image_seq_len, C)
                        timestep=timesteps / 1000,
                        guidance=None,
                        encoder_hidden_states=p_embs,
                        txt_ids=txt_ids,
                        img_ids=img_ids,  # B, image_seq_len, 4
                        joint_attention_kwargs={'attention_mask':attn_mask},
                        return_dict=False,
                    )[0]
    
    batch_size_3_out_diff_padding = batch_size_3_out_diff_padding[:, : latents.size(1) :]
    assert torch.equal(batch_size_3_out, batch_size_3_out_diff_padding), (batch_size_3_out - batch_size_3_out_diff_padding).abs().max()

    # variant to latents
    batch_size_3_out_diff_latents = transformer(
                        hidden_states=diff_latents,  # (B, image_seq_len, C)
                        timestep=timesteps / 1000,
                        guidance=None,
                        encoder_hidden_states=p_embs,
                        txt_ids=txt_ids,
                        img_ids=img_ids,  # B, image_seq_len, 4
                        joint_attention_kwargs={'attention_mask':attn_mask},
                        return_dict=False,
                    )[0]
    batch_size_3_out_diff_padding = batch_size_3_out_diff_padding[:, : latents.size(1) :]
    assert not torch.equal(batch_size_3_out_diff_latents, batch_size_3_out_diff_padding), (batch_size_3_out - batch_size_3_out_diff_padding).abs().max()


def test_dinoscore():
    from utils.eval_utils import pil_to_n1_1_tensor, get_dinoscore

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

# test_dinoscore()

test_attention_mask_and_batched_rope()

print('''\n\n
***********************************
Our tests passed
***********************************
''')


