

###########################################
'''
python src/teacher_generate.py
'''
###########################################


import torch
torch.set_float32_matmul_precision('high')
import logging
import os
import numpy as np
from tqdm import tqdm
from diffusers import Flux2KleinPipeline

from data import get_dataloader
from config import main_config

logging.basicConfig(level=logging.INFO)

LAST_INFERENCE_LATENTS_TIMESTEPS_NOISE_PRED_TUPLES_LIST = []

def collect_latents_on_step_end(self, step: int, timestep: int,
                callback_kwargs):
    global LAST_INFERENCE_LATENTS_TIMESTEPS_NOISE_PRED_TUPLES_LIST
    latents = callback_kwargs['latents']
    noise_pred = callback_kwargs['noise_pred']
    LAST_INFERENCE_LATENTS_TIMESTEPS_NOISE_PRED_TUPLES_LIST.append((latents, timestep, noise_pred))
    return callback_kwargs


def process_batch(pipe, batch, config, ):
    global LAST_INFERENCE_LATENTS_TIMESTEPS_NOISE_PRED_TUPLES_LIST
    assert len(batch['pil_images']) == 1, f'batch size must be 1 not {config.batch_size} for {batch}'
    image = batch['pil_images']
    path = batch['image_paths'][0]
    w, h = image[0].width, image[0].height

    dir = './klein_latents_stimuli/'+os.path.dirname(path)
    filename = os.path.basename(path)

    last_latent_path_example = f'{dir}/{filename}_noise_pred_{3}.pt'

    if os.path.exists(last_latent_path_example):
        return

    logging.info(path)
    
    initial_latent_to_save, _ = pipe.prepare_latents(
        1, pipe.vae.config.latent_channels, h, w,
        torch.bfloat16, 'cuda', generator=torch.Generator(device='cuda')
    )

    # we must undo the packing
    # done later in the pipeline leading to:  # [B, C, H, W] -> [B, H*W, C]

    # we don't care if this viewing is wrong 
    # because it's all random; the shape just needs to be correct.
    initial_latent = initial_latent_to_save.permute(0, 2, 1).view(initial_latent_to_save.shape[0],
                                                          # take dit patch size into account
                                                          pipe.vae.config.latent_channels*4,
                                                          (int(h) // (pipe.vae_scale_factor * 2)),
                                                          (int(w) // (pipe.vae_scale_factor * 2)),
                                                          )

    # we add our initial latent & will backtrack to correctly link the i/o
    #   the final latent will be discarded as it is stepped to produce our clean image.
    LAST_INFERENCE_LATENTS_TIMESTEPS_NOISE_PRED_TUPLES_LIST.append((initial_latent_to_save, None, None))
    pipe._callback_tensor_inputs = ["latents", "prompt_embeds", "noise_pred"]

    im = pipe(image=image, prompt='Generate the image exactly as it was provided.', 
            width=w, height=h, num_inference_steps=4,
            latents=initial_latent,
            callback_on_step_end=collect_latents_on_step_end, 
            callback_on_step_end_tensor_inputs=['latents', 'noise_pred'])[0][0]
    os.makedirs(dir, exist_ok=True)

    tensors_to_save_tuples = []
    for ind, tup in enumerate(LAST_INFERENCE_LATENTS_TIMESTEPS_NOISE_PRED_TUPLES_LIST[:-1]):
        # latent is always the stepped version that will serve as input to the next
        l, _, _ = tup
        _, t, n = LAST_INFERENCE_LATENTS_TIMESTEPS_NOISE_PRED_TUPLES_LIST[ind+1]
        assert not any([a is None for a in [l,t,n]]), f'{[[l,t,n]]=}'
        assert l.numel() == n.numel(), f'{[l.shape, n.shape]=}'
        tensors_to_save_tuples.append((l, t, n))

    for lat_ind, (lat, t, n) in enumerate(tensors_to_save_tuples):
        latent_path = f'{dir}/{filename}_latent_{lat_ind}.pt'
        torch.save(lat, latent_path)
        torch.save(t, f'{dir}/{filename}_timestep_{lat_ind}.pt')
        torch.save(n, f'{dir}/{filename}_noise_pred_{lat_ind}.pt')

        logging.info(f'Latent saved to {latent_path}')

    LAST_INFERENCE_LATENTS_TIMESTEPS_NOISE_PRED_TUPLES_LIST = []


def main(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
                                              torch_dtype=torch.bfloat16
                                              ).to('cuda')

    dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_split_ratio,
                                                 config.batch_size, config.num_workers, config.seed,
                                                 config.resolution, False)
    pipe.transformer = torch.compile(pipe.transformer)
    pipe.vae = torch.compile(pipe.vae)

    for ind, batch in tqdm(enumerate(iter(dataloader))):
        process_batch(pipe, batch, config,)

    for ind, batch in tqdm(enumerate(iter(val_dataloader))):
        process_batch(pipe, batch, config,)



if __name__ == '__main__':
    main(main_config)



