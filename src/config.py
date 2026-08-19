import logging
logging.basicConfig(level=logging.INFO)
import os
import json
import inspect
import torch

from dataclasses import dataclass, field, asdict
from copy import deepcopy


@dataclass
class Config:
    ### Model
    # model_path = None
    transformer_model_path = None
    lora_path = 'logs/wyling_Hippo_Tyzine/4000_ckpt/pytorch_lora_weights.safetensors' # '/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/logs/apostatising_Laennec_Phiona/53000_ckpt/pytorch_lora_weights.safetensors'

    seed: int = 13

    #### seems consistently better to keep text encoder; use edit image
    remove_text_encoder: bool = False
    # as opposed to using RoPE to specify points & their sequence
    scanpath_as_edit_image: bool = True
    ####

    lora_rank: int = 128
    sample_teacher: bool = True

    #### seems consistently better to do all t
    just_inf_timesteps: bool = False
    # just_inf_timesteps will automatically already shift, 
    #   so this does nothing if just_inf_timesteps=False
    shift_timesteps_resolution: bool = False
    ####

    quantize_adam: bool = False
    quantize_model: bool = False

    ### Hparams
    batch_size: int = 32
    lr: float = 1e-4
    use_prompt: str = 'The scene.'

    # teacher gives the input image back in most cases
    #   sans instruction
    teacher_use_prompt: str = ''

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

    ### Data
    included_data_subsets: tuple[str] = ('Art', )
    # we use this for excluding specific samples in "Art" here
    #   be aware that other subsets may use these filenames!
    excluded_data_subsets: tuple[str] = ('035.jpg', '079.jpg', '111.jpg', '115.jpg')

    data_path: str = 'trainSet'
    val_data_split_ratio: int = .1
    num_workers: int = 20
    # width & height side lengths
    resolution: tuple[int, int] = (768, 384)

    ### Logging
    exp_name: str = None
    save_path: str = './'
    freq: int = 500 # how often we save/log/etc.

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

            # Filter kwargs to only include valid parameters
            sig = inspect.signature(super().__init__)
            valid_keys = sig.parameters.keys()            
            valid_kwargs = {k: v for k, v in data.items() if k in valid_keys}
            nonviable_kwargs = {k: v for k, v in data.items() if not k in valid_keys}
            logging.warning(
                f"{nonviable_kwargs} are not used in our config, so we're dropping them!")

            config = cls(**valid_kwargs)
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

    assert not (config.quantize_model and config.lora_rank), (
            'Saving LoRAs on quantized models is broken, so would need to patch the fn.')
    assert isinstance(config.included_data_subsets, tuple), ('We are parsing a tuple of strings, not a string'
                '(This warning avoids parsing each character in a subset as its own inclusion string!)')
    assert config.scanpath_as_edit_image, 'We no longer support RoPE for scanpath conditioning'
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


