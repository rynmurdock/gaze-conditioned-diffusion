

###########################################
'''
python src/teacher_generate.py
'''
###########################################


import torch
torch.set_float32_matmul_precision('high')
import logging
import numpy as np
from tqdm import tqdm
from diffusers import Flux2KleinPipeline

from data import get_dataloader
from config import main_config

logging.basicConfig(level=logging.INFO)


def main(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
                                              # full precision weights
                                              torch_dtype=torch.float32
                                              ).to('cpu')

    dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_split_ratio,
                                                 config.batch_size, config.num_workers, config.seed)

    for epoch in range(config.epochs):
        for ind, batch in tqdm(enumerate(iter(dataloader))):
            image = batch['pil_image']
            pipe.transformer = torch.compile(pipe.transformer)
            pipe.vae = torch.compile(pipe.vae)
            pipe(image=image, prompt='Generate the image exactly as it was provided.', 
                 width=512, height=512,
                 num_inference_steps=2)



if __name__ == '__main__':
    main(main_config)



