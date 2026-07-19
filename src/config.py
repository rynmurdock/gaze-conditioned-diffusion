from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    ### Model
    model_path = None
    # model_path = './last_epoch_ckpt'

    remove_text_encoder: bool = False
    scanpath_as_edit_image: bool = True
    lora_rank: int = 32
    sample_teacher: bool = True
    just_inf_timesteps: bool = True

    quantize_adam: bool = False
    quantize_model: bool = False

    ### Hparams
    batch_size: int = 1
    lr: float = 2e-4
    use_prompt: str = ''
    # TODO test how we can make this give the same re structure but not identical results?
    teacher_use_prompt: str = 'Generate this image as it was given.'

    ### Training
    epochs: int = 3000000000000
    max_steps: int = 3000000000000
    max_val_steps: int = 128

    # this seems to break after d5b46746eb7f329c793d65b76a09c96ef9bfdd97
    # likely do to dynamic shapes being borked on some torch versions
    do_compile: bool = False
    device: str = 'cuda:0'
    
    # specifically for *mixed precision*
    dtype: torch.dtype = field(default=torch.bfloat16, repr=False)
    activation_checkpointing: bool = True
    seed: int = 101

    ### Data
    data_path: str = 'trainSet'
    val_data_split_ratio: int = .1
    num_workers: int = 20
    # width & height side lengths
    resolution: tuple[int, int] = (768, 384)

    use_cached_distilled_latents: bool = False

    ### Logging
    save_path: str = './'
    freq: int = 100 # how often we save/log/etc.

def verify_config_validity(config):
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


