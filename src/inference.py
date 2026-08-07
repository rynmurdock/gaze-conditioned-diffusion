
# import torch

# @torch.no_grad()
# def unpack(pipe, latents, latent_height, latent_width, callback_kwargs): 
#     latents = pipe._unpack_latents_with_ids(latents, callback_kwargs['latent_ids'], latent_height // 2, latent_width // 2); return latents

# @torch.no_grad()
# def visualize_x0_callback_fn(pipe, i, timestep, callback_kwargs): 
#     print('here'); latent_height = 2 * (int(callback_kwargs['height']) // (pipe.vae_scale_factor * 2)); latent_width = 2 * (int(callback_kwargs['width']) // (pipe.vae_scale_factor * 2)); torch.set_grad_enabled(False); latents = unpack(pipe, callback_kwargs.get("latents"), latent_height, latent_width, callback_kwargs); model_output = unpack(pipe, callback_kwargs.get("noise_pred"), latent_height, latent_width, callback_kwargs);  x0 = latents - pipe.scheduler.sigmas[i] * model_output; print('x0:', x0.shape, pipe.vae.bn.running_mean.shape); latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(x0.device, x0.dtype); latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(x0.device, x0.dtype); x0 = (x0 - latents_bn_mean) / latents_bn_std; image = pipe.vae.decode(pipe._unpatchify_latents(x0), return_dict=False)[0]; image = pipe.image_processor.postprocess(image, output_type='pil')[0]; image.save(f'{timestep}.png'); print(callback_kwargs, timestep, i); torch.set_grad_enabled(True); return callback_kwargs

# from diffusers import Flux2KleinPipeline
# Flux2KleinPipeline._callback_tensor_inputs = ["latents", "prompt_embeds", "noise_pred", 'latent_ids', 'height', 'width']
# dtype = torch.bfloat16

# pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
#                                           torch_dtype=dtype,).to('cuda')
# from PIL import Image
# # prompt = 'a photo of inside a house with five windows'

# image = pipe(
#     prompt='',
#     height=512,
#     width=512,
#     guidance_scale=1.0,
#     image=Image.open('./im.jpg'),
#     num_inference_steps=4,
#     callback_on_step_end=visualize_x0_callback_fn,
#     callback_on_step_end_tensor_inputs=["latents", 'height', 'width', "prompt_embeds", "noise_pred", 'latent_ids', 'height', 'width'],
# ).images[0]
# image.save('0.png')

import torch
from model import get_model_and_tokenizer
from config import main_config

config = main_config
model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                    config.dtype, config.seed, config.do_compile, config)
model.config.log_dir = './'
with torch.autocast('cuda'):
    for ind in [1, 2, 4, 5, 8]:
        model.do_qual_val(guidance_scale=ind, im_n=ind)
