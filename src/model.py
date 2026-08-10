import math
import torch
import logging
from tqdm import tqdm
from copy import deepcopy

from modeling.pipe_modded_klein import Flux2KleinPipeline
from modeling.image_cfg_pipe_modded_klein import ImageCFGFlux2KleinPipeline, compute_empirical_mu
from modeling.modded_klein import Flux2Transformer2DModel, prepare_image_ids, prepare_latents, get_inf_timesteps
from data import scanpath_over_pil_image

from diffusers import BitsAndBytesConfig
from diffusers.training_utils import compute_density_for_timestep_sampling
import bitsandbytes as bnb
from peft import LoraConfig
from torchvision.transforms import functional as TF

def ids_encode_pad_mask_images(model, images, dtype):
    latents = []
    image_ids = []
    for pil_img in images:
        img_tensor = TF.to_tensor(pil_img) * 2 - 1  # (3, H, W), values in [-1, 1]
        img_tensor = img_tensor.to(model.device, dtype)[None]
        latent = model.pipe._encode_vae_image(img_tensor, None)
        imids = Flux2KleinPipeline._prepare_image_ids([latent]).to(latent.device)
        image_ids.append(imids)
        latents.append(latent)
    padded_latents = torch.nn.utils.rnn.pad_sequence(latents, batch_first=True,).squeeze(1)
    latents_there_mask = torch.nn.utils.rnn.pad_sequence([torch.ones_like(l) for l in latents], 
                                                         batch_first=True, ).squeeze(1) > 0
    image_ids = torch.nn.utils.rnn.pad_sequence(image_ids, batch_first=True).squeeze(1)
    return padded_latents, image_ids, latents_there_mask

def get_loss(model, images, scanpaths, config, 
             scanpath_sans_contents=None, dtype=None,):
    sample_teacher = config.sample_teacher
    scanpath_as_edit_image = config.scanpath_as_edit_image

    dtype = model.dtype if not dtype else dtype
    with torch.no_grad():
        # rng drop out inputs
        # TODO set drop rate into model from config
        zeroing_mask = torch.rand((scanpaths.shape[0], scanpaths.shape[1])) < .3
        # NOTE our stimuli are actually all the same image size
        # so we don't do attention mask / padding to largest
        scanpaths[zeroing_mask] = 0

        # TODO latents_there_mask for attention masking
        #   because x0 & hint are the same size, latents_there_mask can mask attn over
        #   the latents & the hint
        x0, typical_image_ids, latents_there_mask = ids_encode_pad_mask_images(model, 
                                                                       images, model.config.dtype)
        if scanpath_as_edit_image:
            # scanpath_sans_contents is always padded longest
            hint_latents, hint_ids = model.pipe.prepare_image_latents(
                    images=[scanpath_sans_contents[:1]],
                    batch_size=x0.shape[0],
                    generator=torch.Generator(device='cuda'),
                    device=x0.device,
                    dtype=x0.dtype,
                )
            hint_drop_mask = torch.rand((x0.shape[0],)) < .2
            hint_latents[hint_drop_mask] = 0

        x0 = model.pipe._pack_latents(x0)
        noise = torch.randn_like(x0)

        if config.just_inf_timesteps:
            timesteps = get_inf_timesteps(model.pipe.scheduler, x0, num_inference_steps=4, device='cuda',)
            k = torch.randint(0, 4, (noise.shape[0],)).to(x0.device)
            timesteps = timesteps[k]
        else:
            u = compute_density_for_timestep_sampling(
                weighting_scheme='logit_normal',
                batch_size=x0.shape[0],
                logit_mean=0,
                logit_std=1,
            )
            if config.shift_timesteps_resolution:
                mu = compute_empirical_mu(noise.shape[1], 4)
                u = math.exp(mu) / (math.exp(mu) + (1 / u - 1) ** 1)
            indices = (u * model.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = model.noise_scheduler_copy.timesteps[indices].to(device=x0.device)
        sigma = timesteps.view(-1, 1, 1) / 1000
        latents = sigma * noise + (1 - sigma) * x0

        if sample_teacher:
            model.pipe.transformer.disable_lora()
            
            latent_model_input = torch.cat([latents, x0], dim=1).to(model.pipe.transformer.dtype)
            latent_image_ids = torch.cat([typical_image_ids, typical_image_ids], dim=1)
            teacher_noise_pred = model(latent_model_input, 
                       timesteps=timesteps, image_ids=latent_image_ids, 
                       prompt_embeds=model.pipe.cached_teacher_prompt,
                       txt_ids=model.pipe.cached_teacher_txt_ids,
                       )
            teacher_noise_pred = teacher_noise_pred[:, : latents.size(1) :]
            model.pipe.transformer.enable_lora()


    with torch.autocast(device_type='cuda', enabled=True, dtype=dtype):
        if scanpath_as_edit_image:
            latent_model_input = torch.cat([latents, hint_latents], dim=1).to(model.pipe.transformer.dtype)
            latent_image_ids = torch.cat([typical_image_ids, hint_ids], dim=1)

        output = model(latents if not scanpath_as_edit_image else latent_model_input, 
                       timesteps=timesteps, image_ids=latent_image_ids,
                       prompt_embeds=model.pipe.cached_prompt,
                       txt_ids=model.pipe.cached_txt_ids,
                       )

        if scanpath_as_edit_image:
            output = output[:, : latents.size(1) :]

    if not sample_teacher:
        target = noise - x0
    else:
        target = teacher_noise_pred

    output = output.to(torch.float32)
    target = target.to(torch.float32)
    mse_loss = torch.nn.functional.mse_loss(target, output).mean()
    loss = mse_loss

    logging_dict = {'mse_loss': mse_loss.item(),}
    return loss, logging_dict

def get_random_scanpath_cond_im(width, height, generator, ):
        scanpath_xw = torch.randint(0, width, 
                                    (1, 12, 1), generator=generator, device='cuda')
        scanpath_yh = torch.randint(0, height, 
                                    (1, 12, 1), generator=generator, device='cuda')
        scanpath = torch.cat([scanpath_xw, scanpath_yh], -1)
        cond_img = scanpath_over_pil_image(scanpath[0], 
                                           w=width, 
                                           h=height, 
                                           just_path=True)
        return cond_img, scanpath


class Zoo(torch.nn.Module):
    def __init__(self, pipe, device, dtype, seed=0, config=None) -> None:
        super().__init__()
        self.pipe = pipe
        self.seed = seed
        # NOTE: dtype is the mixed dtype; transformer is still in float32
        self.device, self.dtype = device, dtype
        self.config = config

        self.noise_scheduler_copy = deepcopy(pipe.scheduler)

    def forward(self, latents, timesteps, image_ids, prompt_embeds=None, txt_ids=None):
        if prompt_embeds is None:
            prompt_embeds = torch.zeros(latents.shape[0], 1, 7680).to(latents.device, latents.dtype)
            txt_ids = torch.zeros(latents.shape[0], 4).to(latents.device, latents.dtype)
        
        velocity = self.pipe.transformer(
                hidden_states=latents,  # (B, image_seq_len, C)
                timestep=timesteps / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=image_ids,  # B, image_seq_len, 4
                return_dict=False,
        )[0]
        return velocity

    @torch.no_grad()
    def inference(self, cond_image=None, scanpath=None, guidance_scale=1, generator=None):
        assert cond_image or scanpath is not None 
        width, height = self.config.resolution
        offload_vae_back_to_cpu = False
        # infer vae device from the all params
        if any([p.device != torch.device('cuda:0') for p in self.pipe.vae.parameters()]):
            offload_vae_back_to_cpu = True
            self.pipe.vae = self.pipe.vae.to('cuda')
        prompt_embeds = None
        if not self.config.remove_text_encoder and not isinstance(self.config.use_prompt, str):
            prompt_embeds = torch.zeros(1, 1, 7680).to(self.device, self.dtype)
        else:
            prompt_embeds = self.pipe.cached_prompt

        # if we've only given the scanpath but need it as a cond_image
        if self.config.scanpath_as_edit_image and not cond_image:
            cond_image = scanpath_over_pil_image(scanpath, w=width, h=height, just_path=True)

        # we may use cfg on our cond image
        self.pipe.config.is_distilled = False
        image = self.pipe(
                # just smuggling for our image ids
                image=cond_image if self.config.scanpath_as_edit_image else None,
                latents=scanpath if not self.config.scanpath_as_edit_image else None,
                num_inference_steps=4,
                guidance_scale=guidance_scale,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=prompt_embeds,
                height=height,
                width=width,
                generator=generator,
            ).images[0]
        if offload_vae_back_to_cpu:
            self.pipe.vae = self.pipe.vae.to('cpu')
        return image

    @torch.no_grad()
    def do_qual_val(self, guidance_scale=1, im_n=0,):
        latent_seed_generator = torch.Generator(device="cuda").manual_seed(self.seed)
        for ind, this_seed in enumerate([self.seed, self.seed+179]):
            scanpath_generator = torch.Generator(device="cuda").manual_seed(this_seed)
            width, height = self.config.resolution
            cond_img, scanpath = get_random_scanpath_cond_im(width, height, 
                                                             generator=scanpath_generator)
            image = self.inference(cond_img, scanpath, guidance_scale, latent_seed_generator)
            logging.info(f'Saving at {self.config.log_dir}/sans_scanpath-latest_val_{ind}_{im_n}.png')
            image.save(f'{self.config.log_dir}/sans_scanpath-latest_val_{ind}_{im_n}.png')
            image = scanpath_over_pil_image(scanpath[0], image)
            image.save(f'{self.config.log_dir}/latest_val_{ind}_{im_n}.png')

    
    @torch.no_grad()
    def do_quant_val(self, val_dataloader, max_val_steps, dtype):
        logging.info(f'\nRunning validation for max {max_val_steps}\n')
        # fork_rng temporarily isolates changes
        with torch.random.fork_rng():
            # You can change the seed here locally
            torch.manual_seed(self.seed)

            losses = []
            for index, batch in tqdm(enumerate(val_dataloader)):
                if batch is None:
                    continue

                images, scanpaths = batch['pil_images'], batch['scanpaths']
                scanpaths = scanpaths.to(self.device)
                loss, loss_logging_dict = get_loss(self, 
                                               images, scanpaths, config=self.config,
                                               scanpath_sans_contents=batch.get('scanpath_sans_contents'),
                                               )
                losses.append(loss.item())
                if index >= max_val_steps:
                    return sum(losses) / len(losses)
            return sum(losses) / len(losses)

def get_prompt_embeds_txt_ids(pipe, prompt, device, dtype=torch.float32):
    p, t_ids = pipe.encode_prompt(prompt=prompt, device=device,)
    p, t_ids = p.to(device, dtype), t_ids.to(device, dtype)
    return p, t_ids

def add_lora(transformer, rank):
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank, 
        init_lora_weights="gaussian",
        target_modules='all-linear'
        # could train just the attention for image
        # target_modules=['to_q', 'to_k', 'to_v', 'to_qkv'],
        # exclude_modules=['add_',]
    )
    transformer.add_adapter(transformer_lora_config)
    print(f"trainable params: {transformer.num_parameters(only_trainable=True)} || all params: {transformer.num_parameters()}")

@torch.no_grad()
def get_model_and_tokenizer(path, device, dtype, seed, do_compile, config):
    global Flux2KleinPipeline
    if not config.remove_text_encoder:
        # we can use the vanilla setup besides our prepare_image_ids in this case
        from diffusers import Flux2Transformer2DModel, Flux2KleinPipeline
        # we smuggle in our image ids by overriding both of these & calling scanpath "latents"...
    
    if not config.scanpath_as_edit_image:
        Flux2KleinPipeline.prepare_image_ids = prepare_image_ids
        Flux2KleinPipeline.prepare_latents = prepare_latents

    transformer = Flux2Transformer2DModel.from_pretrained("black-forest-labs/FLUX.2-klein-4B" if path is None
                                                           else path, # we save without a subdir
                                                           subfolder=None if path else 'transformer',
                                                           quantization_config=BitsAndBytesConfig(load_in_8bit=True,) if config.quantize_model else None,
                                                           strict=False)
    if config.lora_path:
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",          # double-stream attention
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",  # double-stream cross/context attention
            "to_qkv_mlp_proj",                            # single-stream fused qkv+mlp-in
            "to_out",                                     # single-stream fused attn-out+mlp-out
        ]
        transformer.load_lora_adapter(f'{config.lora_path}',
                                      prefix=None,
                                      adapter_name='default',
                                      target_modules=target_modules
                                      )
        transformer.set_adapters('default', 1)
        
    elif config.lora_rank:
        # we need a new lora as we aren't loading one
        # inplace operation
        add_lora(transformer, config.lora_rank)

    if config.batch_size > 1:
        from modeling.klein_batched_rope import batchify_transformer_rope
        transformer = batchify_transformer_rope(transformer)

    pipe = ImageCFGFlux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
                                              transformer=transformer,
                                              # full precision weights
                                              torch_dtype=torch.float32,
                                              # we'll put things onto cuda ourselves
                                              device='cpu'
                                              ).to('cpu')

    if config.activation_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

    pipe.vae = pipe.vae.to(device, dtype)
    if do_compile:
        pipe.vae = torch.compile(pipe.vae)
    assert not any([p.device != torch.device('cuda:0') for p in pipe.vae.parameters()]), [n for n, p in pipe.vae.named_parameters() if p.device != torch.device('cuda:0')]

    pipe.cached_prompt, pipe.cached_txt_ids = None, None
    pipe.cached_teacher_prompt, pipe.cached_teacher_txt_ids = None, None
    # NOTE we don't condition on text here
    if isinstance(config.use_prompt, str):
        logging.info('Caching prompt for our model.')
        pipe.text_encoder = pipe.text_encoder.to(device, dtype)
        pipe.cached_prompt, pipe.cached_txt_ids = get_prompt_embeds_txt_ids(pipe, 
                                                                            config.use_prompt, 
                                                                            config.device,)
    if isinstance(config.teacher_use_prompt, str):
        logging.info('Caching prompt for our teacher.')
        # load up the text encoder if we haven't already
        if not isinstance(config.use_prompt, str):
            pipe.text_encoder = pipe.text_encoder.to(config.device)
        pipe.cached_teacher_prompt, pipe.cached_teacher_txt_ids = get_prompt_embeds_txt_ids(pipe,
                                                                                            config.teacher_use_prompt,
                                                                                            config.device,)
    del pipe.text_encoder
    torch.cuda.empty_cache()

    pipe.transformer = pipe.transformer.to(device)

    if do_compile:
        pipe.transformer = torch.compile(pipe.transformer)
    
    model = Zoo(pipe, config.device, config.dtype, seed, config=config).to(device)
    return model

def get_optimizer_and_lr_sched(params, lr, config):
    if config.quantize_adam:
        optimizer = bnb.optim.Adam8bit(params, lr=lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=1)
    return optimizer, scheduler
