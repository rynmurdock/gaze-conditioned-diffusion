from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # Model
    model_path = None
    # model_path = './last_epoch_ckpt'
    lora_rank: int = 16
    quantize_adam: bool = False
    quantize_model: bool = False
    remove_text_encoder: bool = False

    # Hparams
    batch_size: int = 1
    lr: float = 1e-4

    # Training
    epochs: int = 3000000000000
    max_steps: int = 3000000000000
    max_val_steps: int = 128

    do_compile: bool = True
    device: str = 'cuda:0'
    
    # specifically for *mixed precision*
    dtype: torch.dtype = field(default=torch.bfloat16, repr=False)
    activation_checkpointing: bool = True
    seed: int = 101

    # Data
    data_path: str = 'trainSet'
    val_data_split_ratio: int = .1
    num_workers: int = 20
    # width & height side lengths
    resolution: tuple[int, int] = (512, 512)

    use_distilled_latents: bool = True

    # Logging
    save_path: str = './'
    freq: int = 100 # how often we save/log/etc.


main_config = Config()