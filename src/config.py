from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    ### Model
    model_path = None
    # model_path = './last_epoch_ckpt'

    remove_text_encoder: bool = False
    lora_rank: int = 8
    sample_teacher: bool = True

    quantize_adam: bool = False
    quantize_model: bool = False


    ### Hparams
    batch_size: int = 1
    lr: float = 1e-4
    use_prompt: str = 'The scene.'
    teacher_use_prompt: str = 'Generate the image as it was provided.'

    ### Training
    epochs: int = 3000000000000
    max_steps: int = 3000000000000
    max_val_steps: int = 64

    # this seems to occasionally fail out of the blue after d5b46746eb7f329c793d65b76a09c96ef9bfdd97
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


