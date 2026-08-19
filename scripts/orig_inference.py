'''
Inference over a typical klein model with visualization of x0 prediction
python scripts/orig_inference.py
'''

import torch

@torch.no_grad()
def unpack(pipe, latents, latent_height, latent_width, callback_kwargs): 
    latents = pipe._unpack_latents_with_ids(latents, callback_kwargs['latent_ids'], latent_height // 2, latent_width // 2); return latents

@torch.no_grad()
def visualize_x0_callback_fn(pipe, i, timestep, callback_kwargs): 
    print('here'); latent_height = 2 * (int(callback_kwargs['height']) // (pipe.vae_scale_factor * 2)); latent_width = 2 * (int(callback_kwargs['width']) // (pipe.vae_scale_factor * 2)); torch.set_grad_enabled(False); latents = unpack(pipe, callback_kwargs.get("latents"), latent_height, latent_width, callback_kwargs); model_output = unpack(pipe, callback_kwargs.get("noise_pred"), latent_height, latent_width, callback_kwargs);  x0 = latents - pipe.scheduler.sigmas[i] * model_output; print('x0:', x0.shape, pipe.vae.bn.running_mean.shape); latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(x0.device, x0.dtype); latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(x0.device, x0.dtype); x0 = (x0 - latents_bn_mean) / latents_bn_std; image = pipe.vae.decode(pipe._unpatchify_latents(x0), return_dict=False)[0]; image = pipe.image_processor.postprocess(image, output_type='pil')[0]; image.save(f'{timestep}.png'); print(callback_kwargs, timestep, i); torch.set_grad_enabled(True); return callback_kwargs

from diffusers import Flux2KleinPipeline
Flux2KleinPipeline._callback_tensor_inputs = ["latents", "prompt_embeds", "noise_pred", 'latent_ids', 'height', 'width']
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
                                          torch_dtype=dtype,).to('cuda')
from PIL import Image
# prompt = 'a photo of inside a house with five windows'

image = pipe(
    prompt='',
    height=512,
    width=512,
    guidance_scale=1.0,
    image=Image.open('./im.jpg'),
    num_inference_steps=4,
    callback_on_step_end=visualize_x0_callback_fn,
    callback_on_step_end_tensor_inputs=["latents", 'height', 'width', "prompt_embeds", "noise_pred", 'latent_ids', 'height', 'width'],
).images[0]
image.save('0.png')


# import matplotlib.pyplot as plt
# seq_len = 768//16 * 384//16

# # Copied from diffusers.pipelines.flux2.pipeline_flux2.compute_empirical_mu
# def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
#     a1, b1 = 8.73809524e-05, 1.89833333
#     a2, b2 = 0.00016927, 0.45666666

#     if image_seq_len > 4300:
#         mu = a2 * image_seq_len + b2
#         return float(mu)

#     m_200 = a2 * image_seq_len + b2
#     m_10 = a1 * image_seq_len + b1

#     a = (m_200 - m_10) / 190.0
#     b = m_200 - 200.0 * a
#     mu = a * num_steps + b

#     return float(mu)


# import math
# import numpy as np
# mu = compute_empirical_mu(seq_len, num_steps=4)
# u = np.linspace(0, 1, 1000)
# u = math.exp(mu) / (math.exp(mu) + (1 / u - 1) ** 1)
# fig = plt.figure()
# f = plt.plot(u)
# fig.savefig('l.png')
