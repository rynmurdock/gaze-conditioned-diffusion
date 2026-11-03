'''
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python asso_works/compose_paths.py
'''

import os
import sys
import torch
import logging

sys.path.insert(0, os.path.abspathq(os.path.join(os.path.dirname(__file__), '../src')))

from model import get_model_and_tokenizer
from config import main_config
from data import scanpath_over_pil_image

def get_scanpath_points(n_foveations, w, h, generator=None):
    scanpath_xw = torch.randint(0, w, (n_foveations, 1), device='cuda',
                                generator=generator)
    scanpath_yh = torch.randint(0, h, (n_foveations, 1), device='cuda',
                                generator=generator)
    scanpath = torch.cat([scanpath_xw, scanpath_yh], -1)
    return scanpath

def main():
    config = main_config
    config.do_compile = False
    model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                        config.dtype, config.seed, config.do_compile, config)
    model.config.log_dir = './'
    model.dtype = torch.bfloat16
    model.pipe = model.pipe.to(torch.bfloat16)

    width = config.resolution[0] 
    height = config.resolution[1]
    scanpath = get_scanpath_points(0, w=width, h=height)
    latent_seed = 11
    scanpath_generator = torch.Generator('cuda').manual_seed(40)

    # i is how many to add after
    for i, add in enumerate([1, 4, 8, 8]):
        latent_generator = torch.Generator('cuda').manual_seed(latent_seed)

        scanpath_im = scanpath_over_pil_image(scanpath=scanpath, w=width, h=height)
        with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            pred_image = model.inference(guidance_scale=4, 
                                    cond_image=scanpath_im,
                                    generator=latent_generator,
                                    )
            pred_image = scanpath_over_pil_image(scanpath, pred_image)
        path = f'./scratch/{i}.png'
        pred_image.save(path)
        logging.info(' ' + path)

        new_points = get_scanpath_points(add, w=width, h=height, 
                                            generator=scanpath_generator)
        scanpath = torch.cat([scanpath, new_points], 0)


if __name__ == "__main__":
    main()

