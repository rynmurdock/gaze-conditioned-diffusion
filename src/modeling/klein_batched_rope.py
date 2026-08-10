import torch
import types
from torch import nn

from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device
from diffusers.models.embeddings import (
    get_1d_rotary_pos_embed,
)
from diffusers.utils import apply_lora_scale
from diffusers.models.transformers.transformer_flux2 import (_blend_single_block_mods, 
                                                             _blend_double_block_mods,
                                                             Flux2KVCache,
                                                             Flux2Transformer2DModelOutput
                                                             )

def vectorized_batch_apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [B, S, D]
        if sequence_dim == 2:
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]
        elif sequence_dim == 1:
            cos = cos[:, :, None, :]
            sin = sin[:, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


# override in model class
class BatchedFlux2PosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, b_ids: torch.Tensor) -> torch.Tensor:
        # Expected b_ids shape: [B, S, len(self.axes_dim)]
        to_out_sin, to_out_cos  = [], []
        for ids in b_ids:
            cos_out = []
            sin_out = []
            pos = ids.float()
            freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, ids.device)
            # Unlike Flux 1, loop over len(self.axes_dim) rather than ids.shape[-1]
            for i in range(len(self.axes_dim)):
                cos, sin = get_1d_rotary_pos_embed(
                    self.axes_dim[i],
                    pos[..., i],
                    theta=self.theta,
                    repeat_interleave_real=True,
                    use_real=True,
                    freqs_dtype=freqs_dtype,
                )
                cos_out.append(cos)
                sin_out.append(sin)
            freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
            freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
            to_out_sin.append(freqs_sin)
            to_out_cos.append(freqs_cos)
    
        return torch.stack(to_out_cos), torch.stack(to_out_sin)


# patch forward & __call__
@apply_lora_scale("joint_attention_kwargs")
def batch_rope_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs = None,
    return_dict: bool = True,
    kv_cache: "Flux2KVCache | None" = None,
    kv_cache_mode: str | None = None,
    num_ref_tokens: int = 0,
    ref_fixed_timestep: float = 0.0,
) -> torch.Tensor | Flux2Transformer2DModelOutput:
    """
    The [`Flux2Transformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        timestep (`torch.LongTensor`):
            Used to indicate denoising step.
        img_ids (`torch.Tensor`):
            Image position ids used to compute the rotary positional embeddings.
        txt_ids (`torch.Tensor`):
            Text position ids used to compute the rotary positional embeddings.
        guidance (`torch.Tensor`, *optional*):
            Guidance scale embedding used for guidance-distilled variants of the model.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.
        kv_cache (`Flux2KVCache`, *optional*):
            KV cache for reference image tokens. When `kv_cache_mode` is "extract", a new cache is created and
            returned. When "cached", the provided cache is used to inject ref K/V during attention.
        kv_cache_mode (`str`, *optional*):
            One of "extract" (first step with ref tokens) or "cached" (subsequent steps using cached ref K/V). When
            `None`, standard forward pass without KV caching.
        num_ref_tokens (`int`, defaults to `0`):
            Number of reference image tokens prepended to `hidden_states` (only used when
            `kv_cache_mode="extract"`).
        ref_fixed_timestep (`float`, defaults to `0.0`):
            Fixed timestep for reference token modulation (only used when `kv_cache_mode="extract"`).

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor. When `kv_cache_mode="extract"`, also returns the
        populated `Flux2KVCache`.
    """
    num_txt_tokens = encoder_hidden_states.shape[1]

    # 1. Calculate timestep embedding and modulation parameters
    timestep = timestep.to(hidden_states.dtype) * 1000

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)

    # KV extract mode: create cache and blend modulations for ref tokens
    if kv_cache_mode == "extract" and num_ref_tokens > 0:
        num_img_tokens = hidden_states.shape[1]  # includes ref tokens

        kv_cache = Flux2KVCache(
            num_double_layers=len(self.transformer_blocks),
            num_single_layers=len(self.single_transformer_blocks),
        )
        kv_cache.num_ref_tokens = num_ref_tokens

        # Ref tokens use a fixed timestep for modulation
        ref_timestep = torch.full_like(timestep, ref_fixed_timestep * 1000)
        ref_temb = self.time_guidance_embed(ref_timestep, guidance)

        ref_double_mod_img = self.double_stream_modulation_img(ref_temb)
        ref_single_mod = self.single_stream_modulation(ref_temb)

        # Blend double block img modulation: [ref_mod, img_mod]
        double_stream_mod_img = _blend_double_block_mods(
            double_stream_mod_img, ref_double_mod_img, num_ref_tokens, num_img_tokens
        )

    # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    image_rotary_emb = self.pos_embed(img_ids)
    # our text ids are bs=1
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0].expand(image_rotary_emb[0].shape[0], -1, -1), image_rotary_emb[0]], dim=1),
        torch.cat([text_rotary_emb[1].expand(image_rotary_emb[1].shape[0], -1, -1), image_rotary_emb[1]], dim=1),
    )

    # 4. Build joint_attention_kwargs with KV cache info
    if kv_cache_mode == "extract":
        kv_attn_kwargs = {
            **(joint_attention_kwargs or {}),
            "kv_cache": None,
            "kv_cache_mode": "extract",
            "num_ref_tokens": num_ref_tokens,
        }
    elif kv_cache_mode == "cached" and kv_cache is not None:
        kv_attn_kwargs = {
            **(joint_attention_kwargs or {}),
            "kv_cache": None,
            "kv_cache_mode": "cached",
            "num_ref_tokens": kv_cache.num_ref_tokens,
        }
    else:
        kv_attn_kwargs = joint_attention_kwargs

    # 5. Double Stream Transformer Blocks
    for index_block, block in enumerate(self.transformer_blocks):
        if kv_cache_mode is not None and kv_cache is not None:
            kv_attn_kwargs["kv_cache"] = kv_cache.get_double(index_block)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                concat_rotary_emb,
                kv_attn_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=kv_attn_kwargs,
            )

    # Concatenate text and image streams for single-block inference
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # Blend single block modulation for extract mode: [txt_mod, ref_mod, img_mod]
    if kv_cache_mode == "extract" and num_ref_tokens > 0:
        total_single_len = hidden_states.shape[1]
        single_stream_mod = _blend_single_block_mods(
            single_stream_mod, ref_single_mod, num_txt_tokens, num_ref_tokens, total_single_len
        )

    # Build single-block KV kwargs (single blocks need num_txt_tokens)
    if kv_cache_mode is not None:
        kv_attn_kwargs_single = {**kv_attn_kwargs, "num_txt_tokens": num_txt_tokens}
    else:
        kv_attn_kwargs_single = kv_attn_kwargs

    # 6. Single Stream Transformer Blocks
    for index_block, block in enumerate(self.single_transformer_blocks):
        if kv_cache_mode is not None and kv_cache is not None:
            kv_attn_kwargs_single["kv_cache"] = kv_cache.get_single(index_block)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                None,
                single_stream_mod,
                concat_rotary_emb,
                kv_attn_kwargs_single,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=kv_attn_kwargs_single,
            )

    # Remove text tokens (and ref tokens in extract mode) from concatenated stream
    if kv_cache_mode == "extract" and num_ref_tokens > 0:
        hidden_states = hidden_states[:, num_txt_tokens + num_ref_tokens :, ...]
    else:
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

    # 7. Output layers
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if kv_cache_mode == "extract":
        if not return_dict:
            return (output, kv_cache)
        return Flux2Transformer2DModelOutput(sample=output, kv_cache=kv_cache)

    if not return_dict:
        return (output,)

    return Flux2Transformer2DModelOutput(sample=output)



def batchify_transformer_rope(transformer, ):
    import diffusers.models.transformers.transformer_flux2 as flux2_mod
    # orig version of the fn
    _real_apply_rotary_emb = flux2_mod.apply_rotary_emb
    def patched_apply_rotary_emb(x, freqs_cis, sequence_dim=2):
        return vectorized_batch_apply_rotary_emb(x, freqs_cis, sequence_dim=sequence_dim)
    flux2_mod.apply_rotary_emb = patched_apply_rotary_emb
    transformer.forward = types.MethodType(batch_rope_forward, transformer)
    transformer.__call__ = types.MethodType(batch_rope_forward, transformer)
    transformer.pos_embed = BatchedFlux2PosEmbed(transformer.config.rope_theta,
                                                 transformer.config.axes_dims_rope)
    return transformer
