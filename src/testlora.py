import os
import logging
import torch
import sys

from copy import deepcopy
from diffusers import Flux2Transformer2DModel

sys.path.append('/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/src/')

from model import add_lora


def _fresh_transformers():
    orig_transformer = Flux2Transformer2DModel.from_config(
        Flux2Transformer2DModel.load_config("black-forest-labs/FLUX.2-klein-4B", subfolder='transformer')
    )
    intermediate_transformer = deepcopy(orig_transformer)
    add_lora(intermediate_transformer, rank=16)
    return orig_transformer, intermediate_transformer


def _count_lora_params(model):
    return sum(1 for n, _ in model.named_parameters() if 'lora' in n)


def test_root_cause_confirmed_default_prefix_drops_lora():
    """
    Reproduces the failure exactly as in the original script and confirms *why*
    it fails: load_lora_adapter defaults to prefix="transformer", but
    save_lora_adapter (called directly on the transformer, not a pipeline)
    writes unprefixed keys. The filtered state dict is empty, so nothing gets
    injected -- silently (just a logger.warning, no exception).
    """
    orig_transformer, intermediate_transformer = _fresh_transformers()
    tmp_lora_path = '/tmp/tmp_lora_default_prefix/'
    intermediate_transformer.save_lora_adapter(tmp_lora_path, safe_serialization=False)

    caplog_records = []
    handler = logging.Handler()
    handler.emit = lambda record: caplog_records.append(record.getMessage())
    diffusers_logger = logging.getLogger("diffusers.loaders.peft")
    diffusers_logger.addHandler(handler)
    try:
        orig_transformer.load_lora_adapter(tmp_lora_path)  # default prefix="transformer"
    finally:
        diffusers_logger.removeHandler(handler)

    lora_ls = _count_lora_params(orig_transformer)
    assert lora_ls == 0, "Expected the known-bad default-prefix path to inject nothing"
    assert any("No LoRA keys associated" in msg for msg in caplog_records), (
        "Expected diffusers to log its 'no keys found' warning -- if this stops "
        "firing, the library's behavior here has changed and this test should "
        "be revisited."
    )


def test_prefix_none_alone_still_has_adapter_name_mismatch():
    """
    prefix=None fixes the injection, but a *second*, independent bug remains:
    load_lora_adapter doesn't default adapter_name to "default" -- it calls
    get_adapter_name(self), which returns "default_0" for any model with no
    existing PEFT layers, regardless of what name was used when the adapter
    was created/saved. So param names come back as e.g.
    "...lora_A.default_0.weight" instead of "...lora_A.default.weight",
    and `assert n == n1` fails even though the values would have matched.
    """
    orig_transformer, intermediate_transformer = _fresh_transformers()
    tmp_lora_path = '/tmp/tmp_lora_adapter_name_mismatch/'
    intermediate_transformer.save_lora_adapter(tmp_lora_path, safe_serialization=False)

    orig_transformer.load_lora_adapter(tmp_lora_path, prefix=None)  # adapter_name not passed

    orig_names = {n for n, _ in orig_transformer.named_parameters() if 'lora' in n}
    inter_names = {n for n, _ in intermediate_transformer.named_parameters() if 'lora' in n}
    assert orig_names, "sanity check: prefix=None should have injected something"
    assert orig_names != inter_names
    assert any(name.replace('default_0', 'default') in inter_names for name in orig_names), (
        "expected the mismatch to be exactly the default -> default_0 renaming"
    )


def test_fix_prefix_none_and_adapter_name_restores_lora_params():
    """
    The full fix: prefix=None (see above) *and* an explicit, matching
    adapter_name so param names line up exactly with the source adapter.
    "default" is peft's standard adapter name unless add_lora specifies
    otherwise -- check your add_lora implementation if this doesn't match.
    """
    orig_transformer, intermediate_transformer = _fresh_transformers()
    tmp_lora_path = '/tmp/tmp_lora_fix_prefix_none/'
    intermediate_transformer.save_lora_adapter(
        tmp_lora_path, safe_serialization=False, adapter_name="default"
    )

    orig_transformer.load_lora_adapter(tmp_lora_path, prefix=None, adapter_name="default")

    lora_ls = 0
    for (n, p), (n1, p1) in zip(orig_transformer.named_parameters(), intermediate_transformer.named_parameters()):
        if 'lora' in n:
            lora_ls += 1
            assert n == n1, f'{n1, n}'
            assert torch.equal(p, p1), f'{p1, p}'  # NOTE: torch.equal, not torch.equals
    assert lora_ls > 0


def test_fix_is_independent_of_serialization_format():
    """
    Confirms the bug is about the prefix filter, not about safe_serialization
    (.bin vs .safetensors). Uses the default safe_serialization=True path.
    """
    orig_transformer, intermediate_transformer = _fresh_transformers()
    tmp_lora_path = '/tmp/tmp_lora_safetensors/'
    intermediate_transformer.save_lora_adapter(tmp_lora_path)  # safe_serialization=True default

    saved_files = os.listdir(tmp_lora_path)
    assert any(f.endswith('.safetensors') for f in saved_files)

    orig_transformer.load_lora_adapter(tmp_lora_path, prefix=None)
    assert _count_lora_params(orig_transformer) > 0


def test_alternative_manual_peft_state_dict_roundtrip():
    """
    A prefix-agnostic alternative that sidesteps load_lora_adapter's prefix
    filtering entirely by using peft's state-dict helpers directly. Useful if
    you want to decouple from diffusers' pipeline-oriented prefix convention,
    or if add_lora() needs to be called on orig_transformer before loading
    (e.g. because the adapter config -- rank, target_modules -- must be known
    up front rather than inferred from the checkpoint).
    """
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    orig_transformer, intermediate_transformer = _fresh_transformers()

    # Mirror the same LoRA config on orig_transformer so the adapter exists
    # before we try to load weights into it.
    add_lora(orig_transformer, rank=16)

    state_dict = get_peft_model_state_dict(intermediate_transformer)
    incompatible = set_peft_model_state_dict(orig_transformer, state_dict)
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys

    lora_ls = 0
    for (n, p), (n1, p1) in zip(orig_transformer.named_parameters(), intermediate_transformer.named_parameters()):
        if 'lora' in n:
            lora_ls += 1
            assert n == n1
            assert torch.equal(p, p1)
    assert lora_ls > 0


def test_weight_name_explicit_does_not_fix_prefix_bug():
    """
    Guards against a plausible-but-wrong fix: explicitly passing weight_name
    only affects which file is read, not the prefix filter. Confirms prefix
    is the actual culprit, not file discovery.
    """
    orig_transformer, intermediate_transformer = _fresh_transformers()
    tmp_lora_path = '/tmp/tmp_lora_weight_name/'
    intermediate_transformer.save_lora_adapter(
        tmp_lora_path, safe_serialization=False, weight_name="my_lora.bin"
    )

    orig_transformer.load_lora_adapter(tmp_lora_path, weight_name="my_lora.bin")  # still default prefix
    assert _count_lora_params(orig_transformer) == 0

    orig_transformer2, _ = _fresh_transformers()
    orig_transformer2.load_lora_adapter(tmp_lora_path, weight_name="my_lora.bin", prefix=None)
    assert _count_lora_params(orig_transformer2) > 0


def test_diagnose_lora_has_no_effect_on_outputs():
    """
    Diagnostic for "loading succeeds (lora_ls > 0, params match) but the LoRA
    has no effect on the forward pass". Three independent things to rule out,
    in order of likelihood for FLUX.2 specifically:

    1. target_modules coverage: FLUX.2's single-stream blocks
       (Flux2ParallelSelfAttention) have NO to_q/to_k/to_v -- QKV *and* the
       MLP input projection are fused into a single `to_qkv_mlp_proj`
       linear. A target_modules list built around classic
       ["to_q", "to_k", "to_v", "to_out.0"] silently matches nothing in
       every single-stream block. Since those blocks make up a large chunk
       of a FLUX-family transformer, the adapter can end up covering only a
       small slice of the network -- enough for lora_ls > 0, not enough to
       move the output noticeably.
    2. adapter activation: confirm the adapter is actually active post-load.
    3. zero-init weights: if the adapter was never trained (fresh peft
       LoraConfig -> B initialized to zero), *any* correctly-loaded LoRA
       will have exactly zero effect by design. That's expected behavior,
       not a loading bug -- make sure you're testing with trained (or, for
       this test, deliberately perturbed) weights.
    """
    orig_transformer, intermediate_transformer = _fresh_transformers()

    # --- 1. target_modules coverage across block types ---
    linear_module_names = {
        n for n, m in intermediate_transformer.named_modules()
        if isinstance(m, torch.nn.Linear)
    }
    single_stream_targets = {n for n in linear_module_names if "to_qkv_mlp_proj" in n}
    double_stream_targets = {n for n in linear_module_names if n.endswith(("to_q", "to_k", "to_v"))}
    lora_targeted = {
        n.rsplit(".lora_A", 1)[0].rsplit(".lora_B", 1)[0]
        for n, _ in intermediate_transformer.named_parameters() if 'lora' in n
    }
    print(f"single-stream fused qkv+mlp linears: {len(single_stream_targets)}")
    print(f"double-stream q/k/v linears: {len(double_stream_targets)}")
    print(f"linears actually carrying a LoRA adapter: {len(lora_targeted)}")
    assert any('to_qkv_mlp_proj' in n for n in lora_targeted), (
        "add_lora's target_modules does not appear to cover FLUX.2's "
        "single-stream `to_qkv_mlp_proj` layers -- this is the most likely "
        "reason LoRA has negligible effect on outputs. Widen target_modules "
        "(e.g. include 'to_qkv_mlp_proj', 'to_qkv', 'to_added_qkv', "
        "'to_out') or use target_modules='all-linear' if your peft version "
        "supports it."
    )

    # --- 2. adapter is active after load ---
    tmp_lora_path = '/tmp/tmp_lora_diagnose/'
    intermediate_transformer.save_lora_adapter(
        tmp_lora_path, safe_serialization=False, adapter_name="default"
    )
    orig_transformer.load_lora_adapter(tmp_lora_path, prefix=None, adapter_name="default")
    assert "default" in orig_transformer.active_adapters(), (
        "adapter loaded but not active -- call "
        "orig_transformer.set_adapter('default') or "
        "orig_transformer.enable_adapters()"
    )

    # --- 3. rule out zero-init: perturb lora_B so a real effect must exist ---
    with torch.no_grad():
        for n, p in orig_transformer.named_parameters():
            if 'lora_B' in n:
                p.add_(torch.randn_like(p) * 0.5)


if __name__ == "__main__":
    test_root_cause_confirmed_default_prefix_drops_lora()
    print("test_root_cause_confirmed_default_prefix_drops_lora passed")
    test_prefix_none_alone_still_has_adapter_name_mismatch()
    print("test_prefix_none_alone_still_has_adapter_name_mismatch passed")
    test_fix_prefix_none_and_adapter_name_restores_lora_params()
    print("test_fix_prefix_none_and_adapter_name_restores_lora_params passed")
    test_diagnose_lora_has_no_effect_on_outputs()
    print("test_diagnose_lora_has_no_effect_on_outputs passed")
    test_fix_is_independent_of_serialization_format()
    print("test_fix_is_independent_of_serialization_format passed")
    test_alternative_manual_peft_state_dict_roundtrip()
    print("test_alternative_manual_peft_state_dict_roundtrip passed")
    test_weight_name_explicit_does_not_fix_prefix_bug()
    print("test_weight_name_explicit_does_not_fix_prefix_bug passed")