import torch
import sys

from copy import deepcopy
from diffusers import Flux2Transformer2DModel

sys.path.append('/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/src/')

from model import add_lora

def test_lora_save_and_reload():
    orig_transformer = Flux2Transformer2DModel.from_config(Flux2Transformer2DModel.load_config("black-forest-labs/FLUX.2-klein-4B", subfolder='transformer'))
    intermediate_transformer = deepcopy(orig_transformer)
    add_lora(intermediate_transformer, rank=16)
    tmp_lora_path = '/tmp/tmp_l/'
    intermediate_transformer.save_lora_adapter(tmp_lora_path, safe_serialization=False, adapter_name='default')
    orig_transformer.load_lora_adapter(tmp_lora_path, prefix=None, adapter_name='default')

    lora_ls = 0
    for ((n, p), (n1, p1)) in zip(orig_transformer.named_parameters(), intermediate_transformer.named_parameters()):
        if 'lora' in n:
            lora_ls += 1
            assert n == n1, f'{n1, n}'
            assert torch.equal(p, p1), f'{p1, p}'
    assert lora_ls > 0
    print('''
    ****************
    Passed
    ****************''')

# TODO setup pytest
test_lora_save_and_reload()