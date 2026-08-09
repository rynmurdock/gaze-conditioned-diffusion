
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import get_model_and_tokenizer
from config import main_config
from data import get_dataloader
from utils.eval_utils import get_dinoscore, get_lpips


config = main_config
model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                    config.dtype, config.seed, config.do_compile, config)
model.config.log_dir = './'
model.config.seed = 11

__train_dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_split_ratio,
                                                 config.batch_size, config.num_workers, config.seed,
                                                 config.resolution, config.use_cached_distilled_latents)

with torch.autocast('cuda'):
    for ind in [5, 6, 7, 8, 9, 11]:
        model.do_qual_val(guidance_scale=ind, im_n=ind)

# TODO:
#   do we benefit from cfg on image? (start small like 1.2 -- + i doubt it)
#   lpips, image clipscore

