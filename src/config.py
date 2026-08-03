from dataclasses import dataclass, field, asdict
from copy import deepcopy
import logging
logging.basicConfig(level=logging.INFO)
import os
import json
import torch


@dataclass
class Config:
    ### Model
    # model_path = None
    transformer_model_path = None
    lora_path = './logs/Ede_Joelly/10_ckpt/'

    #### seems consistently better to keep text encoder; use edit image
    remove_text_encoder: bool = False
    # as opposed to using RoPE to specify points & their sequence
    scanpath_as_edit_image: bool = True
    ####


    lora_rank: int = 128
    sample_teacher: bool = True

    #### seems consistently better to do all t
    just_inf_timesteps: bool = False
    ####

    quantize_adam: bool = False
    quantize_model: bool = False

    ### Hparams
    batch_size: int = 1
    lr: float = 1e-5
    use_prompt: str = 'The scene that was seen.'
    # TODO test how we can make this give the same re structure but not identical results?
    teacher_use_prompt: str = 'Generate this image as it was given.'

    ### Training
    epochs: int = 3000000000000
    max_steps: int = 100_000
    max_val_steps: int = 64

    # this seems to break after d5b46746eb7f329c793d65b76a09c96ef9bfdd97
    # likely due to dynamic shapes being borked on some torch versions
    do_compile: bool = False
    device: str = 'cuda:0'
    
    # specifically for *mixed precision*
    # we parse torch dtypes to str on saving & then back on loading for simplicity
    dtype: torch.dtype = field(default=torch.bfloat16, repr=False)
    activation_checkpointing: bool = True
    seed: int = 11

    ### Data
    data_path: str = 'trainSet'
    val_data_split_ratio: int = .1
    num_workers: int = 20
    # width & height side lengths
    resolution: tuple[int, int] = (768, 384)

    use_cached_distilled_latents: bool = False

    ### Logging
    exp_name: str = None
    save_path: str = './'
    freq: int = 1000 # how often we save/log/etc.



    def to_json(self, filename):
        # we don't want to mutate our actual class
        cfg = deepcopy(self)

        # e.g. torch.bfloat16 -> bfloat16
        cfg.dtype = str(cfg.dtype).split('.')[-1]
        with open(filename, "w") as f:
            json.dump(asdict(cfg), f)

    @classmethod
    def from_json(cls, filename):
        '''
            An unnecessary class method added solely to horrify non-CS juniors. "o.o.p."s
            Loads a config json file using the config class to make a config object.
        '''

        with open(filename, "r") as file:
            data = json.load(file)
            config = cls(**data)
        config = parse_dtype(config)
        return config

def parse_dtype(config):
    if not isinstance(config.dtype, torch.dtype):
        if isinstance(config.dtype, str):
            try:
                logging.info(f'{config.dtype=}')
                torch_dtype = getattr(torch, config.dtype)
                config.dtype = torch_dtype
            except Exception as e:
                logging.error(f'Error trying to parse dtype: {e}')
                raise(Exception)
        else:
            assert False, f'{config.dtype} is not a torch dtype'
    return config


def verify_config_validity(config):
    parse_dtype(config)

    assert config.batch_size == 1, 'We do not support batch_size > 1 yet.'
    assert not (config.sample_teacher and config.use_cached_distilled_latents), (
        "There's no reason to try to use our cached latents and sample new ones"
    )
    assert not (config.sample_teacher and not (not config.remove_text_encoder and config.lora_rank)), (
        'sample_teacher is only allowed with LoRA and text encoders kept. '
        'we directly turn off our LoRA, grab a random input/output pair, then train on it. '
        'We want our teacher to already be there and undisturbed.'
        )

main_config = Config()

if __name__ == "__main__":
    # TODO use pytest instead
    orig_main_config = Config()
    orig_main_config.to_json('./placeholder_conf.json')
    new_conf = Config.from_json('./placeholder_conf.json')
    assert orig_main_config == new_conf, (
        'Original and reloaded configs are not equal!'
        f'\n{new_conf=}'
        f'\nOriginal: {orig_main_config=}')
    os.remove('./placeholder_conf.json')


