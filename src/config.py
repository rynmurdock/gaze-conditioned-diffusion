from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # Model
    model_path = None
    # model_path = './last_epoch_ckpt'

    # Hparams
    batch_size: int = 1
    lr: float = 1e-5

    # how many scanpaths to max at
    k: int = 8


    # Training
    epochs: int = 3000000000000
    max_steps: int = 400000

    # TODO should be much higher
    max_val_steps: int = 32

    do_compile: bool = True
    device: str = 'cuda:0'
    
    # specifically for *mixed precision*
    dtype: torch.dtype = field(default=torch.bfloat16, repr=False)
    seed: int = 107

    # Data
    data_path: str = 'trainSet'
    val_data_split_ratio: int = .1
    num_workers: int = 8

    # Logging
    save_path: str = './'
    freq: int = 100  # how often we save/log/etc.


main_config = Config()