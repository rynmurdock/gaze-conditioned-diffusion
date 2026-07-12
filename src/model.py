
import torch
import logging

from diffusers import Flux2KleinPipeline
from modded_klein import Flux2Transformer2DModel, prepare_image_ids


def get_loss(model, x0, scanpaths):
    with torch.no_grad():
        # rng drop out inputs
        # TODO set drop rate into model from config
        zeroing_mask = torch.rand((scanpaths.shape[0], scanpaths.shape[1])) < .3
        x0 = model.pipe._pack_latents(model.pipe._encode_vae_image(x0, None))

        # NOTE our stimuli are actually all the same image size
        # so we don't do attention mask / padding to largest
        scanpaths[zeroing_mask] = 0

        noise = torch.randn(x0.shape[0], x0.shape[-1], device=x0.device)
        timestep = torch.randint(0, 1000, (noise.shape[0],)).to(x0.device)

        # TODO use setup of u = from dreambooth
        sigma = timestep / 1000
        latent = sigma * noise + (1 - sigma) * x0
    
    with torch.autocast(device_type='cuda', enabled=True, dtype=model.dtype):
        output = model(latent, scanpaths, 
                       timesteps=timestep,
                       )

    target = noise - x0

    output = output.to(torch.float32)
    target = target.to(torch.float32)
    mse_loss = torch.nn.functional.mse_loss(target, output).mean()
    loss = mse_loss

    logging_dict = {'mse_loss': mse_loss.item(),}
    return loss, logging_dict


class Zoo(torch.nn.Module):
    def __init__(self, pipe, device, dtype, seed=0) -> None:
        super().__init__()
        self.pipe = pipe
        self.seed = seed
        # NOTE: dtype is the mixed dtype; transformer is still in float32
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def do_qual_val(self, images):
        raise(NotImplementedError(
            (
            'We"ll want to override the pipe to give fixations '
             'to Klein"s RoPE'
             )
        ))
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        images = self.pipe(
            scanpath=torch.randint(0, 10, (5,)).to('cuda'),
            num_inference_steps=50,
            guidance_scale=8,
            generator=generator
        ).images
        images[0].save('latest_val.png')
        return images
    
    def forward(self, latents, scanpath, timesteps):
        gaze_image_ids = prepare_image_ids([latents], scanpath)

        # three layers from qwen3 = 3x2560 on actual inner dim
        prompt_embeds = latents.new_zeros(1, 1, 7680)
        logging.warning('We aren"t using scanpath yet!')
        velocity = self.pipe.transformer(
                hidden_states=latents,  # (B, image_seq_len, C)
                timestep=timesteps / 1000,
                guidance=None,
                encoder_hidden_states=None,
                txt_ids=latents.new_zeros(len(latents), 1, 4),  # B, text_seq_len, 4
                img_ids=gaze_image_ids,  # B, image_seq_len, 4
                return_dict=False,
        )[0]
        return velocity

    
    @torch.no_grad()
    def do_quant_val(self, val_dataloader, max_val_steps):
        # fork_rng temporarily isolates changes
        with torch.random.fork_rng():
            # You can change the seed here locally
            torch.manual_seed(self.seed)

            losses = []
            for index, batch in enumerate(val_dataloader):
                if batch is None:
                    continue

                x0, scanpaths = batch['images'], batch['scanpaths']
                print('device',self.device)
                print('device',self.dtype)
                x0 = x0.to(self.device)
                scanpaths = scanpaths.to(self.device)
                loss, loss_logging_dict = get_loss(self, 
                                                x0, scanpaths)
                losses.append(loss.item())
                if index > max_val_steps:
                    return sum(losses) / len(losses)        
            return sum(losses) / len(losses)
    
def get_model_and_tokenizer(path, device, dtype, seed, do_compile, config):    
    transformer = Flux2Transformer2DModel.from_pretrained("black-forest-labs/FLUX.2-klein-4B" if path is None
                                                           else path, subfolder='transformer',
                                                           strict=False)
    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
                                              transformer=transformer,
                                              # full precision weights
                                              torch_dtype=torch.float32
                                              ).to('cpu')
    # TODO assert transformer dtype
    transformer = pipe.transformer.to(device)
    if config.activation_checkpointing:
        transformer.enable_gradient_checkpointing()
    vae = pipe.vae.to(device, dtype)
    # NOTE we don't condition on text here
    del pipe.text_encoder

    if do_compile:
        transformer = torch.compile(transformer)
        vae = torch.compile(vae)
    
    model = Zoo(pipe, config.device, config.dtype, seed).to(device)
    # TODO could set as attributes (idk why I like this option), could just put config into __init__
    model.k = config.k
    
    return model

def get_optimizer_and_lr_sched(params, lr):
    logging.info(f'Training: {params}')
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    return optimizer, scheduler
