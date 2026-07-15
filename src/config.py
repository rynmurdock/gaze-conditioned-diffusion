from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # Model
    model_path = None
    model_path = './last_epoch_ckpt'

    # Hparams
    batch_size: int = 1
    lr: float = 2e-5

    # Training
    epochs: int = 3000000000000
    max_steps: int = 3000000000000

    # TODO should be much higher
    max_val_steps: int = 32

    do_compile: bool = True
    device: str = 'cuda:0'
    
    # specifically for *mixed precision*
    dtype: torch.dtype = field(default=torch.bfloat16, repr=False)
    activation_checkpointing: bool = True
    seed: int = 107

    # Data
    data_path: str = 'trainSet'
    val_data_split_ratio: int = .1
    num_workers: int = 20
    # width & height side lengths
    resolution: int = 512

    use_distilled_latents: bool = True

    # Logging
    save_path: str = './'
    freq: int = 1000  # how often we save/log/etc.


main_config = Config()