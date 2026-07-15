
import torch
import logging

from pipe_modded_klein import Flux2KleinPipeline
from modded_klein import Flux2Transformer2DModel, prepare_image_ids

from types import SimpleNamespace


def get_loss(model, image, scanpaths, 
             latents=None, timesteps=None, noise_pred=None,
             dtype=None):
    dtype = model.dtype if not dtype else dtype
    with torch.no_grad():
        # rng drop out inputs
        # TODO set drop rate into model from config
        zeroing_mask = torch.rand((scanpaths.shape[0], scanpaths.shape[1])) < .3
        # NOTE our stimuli are actually all the same image size
        # so we don't do attention mask / padding to largest
        scanpaths[zeroing_mask] = 0

        # we load pre-encoded images
        if latents is None:
            x0 = model.pipe._encode_vae_image(image, None)
        else:
            latents = latents.to(image.device)
            noise_pred = noise_pred.to(image.device)
            timesteps = timesteps.to(image.device)
            target = noise_pred
            # misnomer but fine for our purposes
            x0 = target.new_zeros(
                image.shape[0],
                # take dit patch size into account
                model.pipe.vae.config.latent_channels*4,
                (int(image.shape[-2]) // (model.pipe.vae_scale_factor * 2)),
                (int(image.shape[-1]) // (model.pipe.vae_scale_factor * 2)),
                )

        gaze_image_ids = prepare_image_ids([x0], scanpaths)
        x0 = model.pipe._pack_latents(x0)

        noise = torch.randn_like(x0)

        if latents is None:
            timesteps = torch.randint(0, 1000, (noise.shape[0],)).to(x0.device)
            # TODO don't use uniform sampling
            sigma = timesteps / 1000
            latents = sigma * noise + (1 - sigma) * x0


    with torch.autocast(device_type='cuda', enabled=True, dtype=dtype):
        output = model(latents, 
                       timesteps=timesteps, gaze_image_ids=gaze_image_ids,
                       )

    if noise_pred is None:
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

    def forward(self, latents, timesteps, gaze_image_ids):
        velocity = self.pipe.transformer(
                hidden_states=latents,  # (B, image_seq_len, C)
                timestep=timesteps / 1000,
                guidance=None,
                encoder_hidden_states=None,
                txt_ids=None,
                img_ids=gaze_image_ids,  # B, image_seq_len, 4
                return_dict=False,
        )[0]
        return velocity

    @torch.no_grad()
    def do_qual_val(self,):
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        images = self.pipe(
            scanpath=torch.randint(0, 10, (1, 5, 2)).to('cuda'),
            num_inference_steps=4,
            guidance_scale=1,
            height=512,
            width=512,
            generator=generator,
        ).images
        images[0].save('latest_val.png')
        return images
    
    @torch.no_grad()
    def do_quant_val(self, val_dataloader, max_val_steps, dtype):
        # fork_rng temporarily isolates changes
        with torch.random.fork_rng():
            # You can change the seed here locally
            torch.manual_seed(self.seed)

            losses = []
            for index, batch in enumerate(val_dataloader):
                if batch is None:
                    continue

                image, scanpaths = batch['images'], batch['scanpaths']
                image = image.to(self.device, dtype)
                scanpaths = scanpaths.to(self.device)
                loss, loss_logging_dict = get_loss(self, 
                                                image, scanpaths, 
                                               latents=batch.get('latents'),
                                               timesteps=batch.get('timesteps'),
                                               noise_pred=batch.get('noise_preds'),
                                               )
                losses.append(loss.item())
                if index > max_val_steps:
                    return sum(losses) / len(losses)
            return sum(losses) / len(losses)

def get_model_and_tokenizer(path, device, dtype, seed, do_compile, config):    
    transformer = Flux2Transformer2DModel.from_pretrained("black-forest-labs/FLUX.2-klein-4B" if path is None
                                                           else path, # we save without a subdir
                                                           subfolder=None if path else 'transformer',
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

    if not config.use_distilled_latents:
        pipe.vae = pipe.vae.to(device, dtype)
        if do_compile:
            pipe.vae = torch.compile(pipe.vae)
        assert not any([p.device != torch.device('cuda:0') for p in pipe.vae.parameters()]), [p for p in pipe.vae.parameters() if p.device != torch.device('cuda:0')]
    else:
        # we can solely put our vae onto cuda only for our qual validation where we need decoding
        pipe.vae = pipe.vae.to(dtype)

    # NOTE we don't condition on text here
    del pipe.text_encoder

    if do_compile:
        transformer = torch.compile(transformer)
    
    model = Zoo(pipe, config.device, config.dtype, seed).to(device)
    return model

def get_optimizer_and_lr_sched(params, lr):
    logging.info(f'Training: {params}')
    optimizer = torch.optim.SGD(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=2, eta_min=8e-7)
    return optimizer, scheduler
