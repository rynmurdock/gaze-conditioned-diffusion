import torch
import logging
from tqdm import tqdm
from copy import deepcopy

from modeling.image_cfg_pipe_modded_klein import ImageCFGFlux2KleinPipeline, compute_empirical_mu, calculate_shift, get_inf_timesteps
from data import scanpath_over_pil_image

from diffusers import BitsAndBytesConfig, Flux2Transformer2DModel
from diffusers.training_utils import compute_density_for_timestep_sampling
import bitsandbytes as bnb
from peft import LoraConfig
from torchvision.transforms import functional as TF

@torch.no_grad()
def unpack(pipe, latents, latent_height, latent_width, callback_kwargs): 
    latents = pipe._unpack_latents_with_ids(latents, callback_kwargs['latent_ids'], latent_height // 2, latent_width // 2); return latents

@torch.no_grad()
def visualize_x0_callback_fn(pipe, i, timestep, callback_kwargs): 
    print('here'); latent_height = 2 * (int(callback_kwargs['height']) // (pipe.vae_scale_factor * 2)); latent_width = 2 * (int(callback_kwargs['width']) // (pipe.vae_scale_factor * 2)); torch.set_grad_enabled(False); latents = unpack(pipe, callback_kwargs.get("latents"), latent_height, latent_width, callback_kwargs); model_output = unpack(pipe, callback_kwargs.get("noise_pred"), latent_height, latent_width, callback_kwargs);  x0 = latents - pipe.scheduler.sigmas[i] * model_output; print('x0:', x0.shape, pipe.vae.bn.running_mean.shape); latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(x0.device, x0.dtype); latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(x0.device, x0.dtype); x0 = (x0 - latents_bn_mean) / latents_bn_std; image = pipe.vae.decode(pipe._unpatchify_latents(x0), return_dict=False)[0]; image = pipe.image_processor.postprocess(image, output_type='pil')[0]; image.save(f'{timestep}.png'); print(callback_kwargs, timestep, i); torch.set_grad_enabled(True); return callback_kwargs


def ids_encode_pad_mask_images(model, images, dtype):
    with torch.autocast(device_type='cuda', enabled=True, dtype=dtype):
        latents = []
        image_ids = []
        latent_ids = []
        for pil_img in images:
            img_tensor = TF.to_tensor(pil_img) * 2 - 1  # (3, H, W), values in [-1, 1]
            img_tensor = img_tensor.to(model.device, dtype)[None]
            latent = model.pipe._encode_vae_image(img_tensor, None)
            # prepare_image_latents != prepare_latent_ids -- 
            #   former gives a shift as they're each a cond image
            imids = ImageCFGFlux2KleinPipeline._prepare_image_ids([latent],).to(latent.device)
            latids = ImageCFGFlux2KleinPipeline._prepare_latent_ids(latent).to(latent.device)
            image_ids.append(imids[0])
            latent_ids.append(latids[0])
            latents.append(model.pipe._pack_latents(latent)[0])
        padded_latents = torch.nn.utils.rnn.pad_sequence(latents, batch_first=True,).squeeze(1)
        latents_there_mask = torch.nn.utils.rnn.pad_sequence([torch.ones_like(l) for l in latents], 
                                                            batch_first=True, ).squeeze(1) > 0
        image_ids = torch.nn.utils.rnn.pad_sequence(image_ids, batch_first=True).squeeze(1)
        latent_ids = torch.nn.utils.rnn.pad_sequence(latent_ids, batch_first=True).squeeze(1)
        return padded_latents, image_ids, latents_there_mask, latent_ids

        # with torch.autocast(device_type='cuda', ):
        #     visualize_x0_callback_fn(model.pipe, t_ind, timesteps[:, t_ind], {
        #         'height': hw[0],
        #         'width': hw[1],
        #         'latent': latents,
        #         'latents': latents,
        #         'noise_pred': teacher_noise_pred,
        #         'latent_ids': latent_image_ids[:,:latent_image_ids.shape[1]//2],
        #     })


def get_loss(model, images, scanpaths, config, 
             scanpath_sans_contents=None, dtype=None,):
    sample_teacher = config.sample_teacher

    dtype = model.dtype if not dtype else dtype
    with torch.no_grad():
        x0, typical_image_ids, latents_there_mask, noisy_image_ids = ids_encode_pad_mask_images(model, 
                                                                       images, model.config.dtype)
        teacher_latent_image_ids = torch.cat([noisy_image_ids, typical_image_ids], dim=1)
        
        hint_latents, hint_ids, _, _ = ids_encode_pad_mask_images(model, 
                                                                    scanpath_sans_contents, model.config.dtype)

        hint_drop_mask = torch.rand((x0.shape[0],)) < .1
        hint_latents[hint_drop_mask] = 0

        hw = (max([a.height for a in images]), max([a.width for a in images]))
        if config.sample_full_trajectory:
            timesteps_set = get_inf_timesteps(model.noise_scheduler_copy, latents_there_mask, 
                                    num_inference_steps=4, device='cuda',)
            # appending zero timesteps
            timesteps_set = torch.nn.functional.pad(timesteps_set, (0, 1))

        else:
            noise = torch.randn_like(x0)
            if config.just_inf_timesteps:
                timesteps = get_inf_timesteps(model.noise_scheduler_copy, latents_there_mask, num_inference_steps=4, device='cuda',)
                k = torch.randint(0, 4, (noise.shape[0],)).to(x0.device)
                timesteps = timesteps[torch.arange(noise.shape[0], device=x0.device), k]
            else:
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.timestep_density_fn if config.timestep_density_fn else 'logit_normal',
                    batch_size=x0.shape[0],
                    logit_mean=0,
                    logit_std=1,
                )
                # shift per sample using its mask for seq len of non-padding
                if config.shift_timesteps_resolution:
                    mus = []
                    # NOTE this ditches compute_empirical_mu
                    for sample_ind in range(noise.shape[0]):
                        mu = calculate_shift(latents_there_mask[sample_ind].amax(-1).sum(0), )
                        mus.append(mu)
                    mus = torch.tensor(mus).to(u.device, u.dtype)
                    u = torch.exp(mus) / (torch.exp(mus) + (1 / u - 1) ** 1)
                indices = (u * model.noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = model.noise_scheduler_copy.timesteps[indices].to(device=x0.device)
            sigma = timesteps.view(-1, 1, 1) / 1000
            latents = sigma * noise + (1 - sigma) * x0

            if sample_teacher:
                model.pipe.transformer.disable_lora()
                
                latent_model_input = torch.cat([latents, x0], dim=1).to(model.pipe.transformer.dtype)
                teacher_noise_pred = model(latent_model_input, 
                        timesteps=timesteps, image_ids=teacher_latent_image_ids, 
                        prompt_embeds=model.pipe.cached_teacher_prompt,
                        txt_ids=model.pipe.cached_teacher_txt_ids,
                        latents_attention_mask=latents_there_mask.repeat(1, 2, 1),
                        )
                teacher_noise_pred = teacher_noise_pred[:, : latents.size(1) :]
                model.pipe.transformer.enable_lora()
        
            # our non-full set are going to have a single nfe
            timesteps_set = timesteps[:, None]
            

    grand_loss = 0
    with torch.autocast(device_type='cuda', enabled=not config.quantize_model, dtype=dtype):
        assert torch.equal(hint_ids, typical_image_ids), (
            f'Should be equal: {hint_ids} != {typical_image_ids}'
        )
        assert not torch.equal(hint_ids, noisy_image_ids), (
            f'Should not be equal: {hint_ids} == {typical_image_ids}'
        )

        for ind in range(timesteps_set.shape[1]):
            if config.sample_full_trajectory:
                if ind == 0: into = torch.randn_like(x0)
                if ind == timesteps_set.shape[1]-1: continue
            else:
                into = inputs[:, ind]; target = targets[:, ind]
                
            latent_model_input = torch.cat([into, hint_latents], dim=1).to(model.pipe.transformer.dtype)
            latent_image_ids = torch.cat([noisy_image_ids, hint_ids], dim=1)

            student_pred = model(latent_model_input, 
                        timesteps=timesteps_set[:, ind,], image_ids=latent_image_ids,
                        prompt_embeds=model.pipe.cached_prompt,
                        txt_ids=model.pipe.cached_txt_ids,
                        latents_attention_mask=latents_there_mask.repeat(1, 2, 1),
                        )
            student_pred = student_pred[:, : into.size(1) :]

            if config.sample_full_trajectory:
                with torch.no_grad():
                    model.pipe.transformer.disable_lora()
                    latent_model_input = torch.cat([into, x0], dim=1).to(model.pipe.transformer.dtype)
                    teacher_noise_pred = model(latent_model_input, 
                                timesteps=timesteps_set[:, ind, ], image_ids=latent_image_ids, 
                                prompt_embeds=model.pipe.cached_teacher_prompt,
                                txt_ids=model.pipe.cached_teacher_txt_ids,
                                latents_attention_mask=latents_there_mask.repeat(1, 2, 1),
                                )
                    teacher_noise_pred = teacher_noise_pred[:, : into.size(1) :]
                    target = into - timesteps_set[:, ind, None, None,] / 1000 * teacher_noise_pred
                    model.pipe.transformer.enable_lora()

            # we do x0 parameterization when we're going over full trajectories
            if config.sample_full_trajectory:
                # would rather diffusers.step but it is a nightmare in there.
                if ind < timesteps_set.shape[1]-1:
                    t_a = timesteps_set[:, ind+1]
                else:
                    t_a = 0
                # x0
                output = into - timesteps_set[:, ind, None, None,] / 1000 * student_pred
                # we step our latent
                into = into + (t_a/1000 - timesteps_set[:, ind]/1000)[:, None, None,] * student_pred

            output = output.to(torch.float32)
            target = target.to(torch.float32)
            loss = (target - output)**2
            # mask anywhere we don't have contents
            loss[~latents_there_mask] = 0
            # mean over batch last
            loss = loss.flatten(1).sum(1) / latents_there_mask.flatten(1).sum(1)
            loss = loss.mean()
            grand_loss += loss
        grand_loss = grand_loss / timesteps_set.shape[1]

    logging_dict = {'mse_loss': grand_loss.item(),}
    return grand_loss, logging_dict

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

    def forward(self, latents, timesteps, image_ids, 
                prompt_embeds=None, txt_ids=None, latents_attention_mask=None):
        # latents_attention_mask: zeroes where padded & ones where contents of shape [B, S, D]
        if self.config.quantize_model:
            latents = latents.to(torch.float16)
            timesteps = timesteps.to(torch.float16)
            prompt_embeds = prompt_embeds.to(torch.float16) if prompt_embeds is not None else None

        if prompt_embeds is None:
            prompt_embeds = torch.zeros(latents.shape[0], 1, 7680).to(latents.device, latents.dtype)
            txt_ids = torch.zeros(latents.shape[0], 4).to(latents.device, latents.dtype)

        # account for text first in sequence; sum for just the [B, L], make boolean
        latents_attention_mask = latents_attention_mask.sum(-1)
        attention_mask = torch.nn.functional.pad(latents_attention_mask, 
                                                 (prompt_embeds.shape[1], 0,), value=1) != 0.
        if len(prompt_embeds) == 1 and len(prompt_embeds) != len(latents):
            prompt_embeds = prompt_embeds.repeat(len(latents), 1, 1)

        velocity = self.pipe.transformer(
                hidden_states=latents,  # (B, image_seq_len, C)
                timestep=timesteps / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=image_ids, # B, image_seq_len, 4
                joint_attention_kwargs={'attention_mask': attention_mask},
                return_dict=False,
        )[0]
        return velocity

    @torch.no_grad()
    def inference(self, 
                  cond_image=None, 
                  scanpath=None, 
                  guidance_scale=1,
                  width_height=None, 
                  generator=None):
        assert cond_image or scanpath is not None
        width, height = self.config.resolution if not width_height else (width_height[0], width_height[1])
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
        if not cond_image:
            cond_image = scanpath_over_pil_image(scanpath, w=width, h=height, just_path=True)

        # we may use cfg on our cond image
        self.pipe.config.is_distilled = False
        image = self.pipe(
                # just smuggling for our image ids
                image=cond_image,
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
    def do_qual_val(self, guidance_scale=2, im_n=0, step_n=None, cond_img=None, scanpath=None):
        latent_seed_generator = torch.Generator(device="cuda").manual_seed(self.seed)
        for ind, this_seed in enumerate([self.seed, self.seed+179]):
            scanpath_generator = torch.Generator(device="cuda").manual_seed(this_seed)
            width, height = self.config.resolution
            if not cond_img:
                cond_img, scanpath = get_random_scanpath_cond_im(width, height, 
                                                             generator=scanpath_generator)
            image = self.inference(cond_img, scanpath, guidance_scale, (width, height), latent_seed_generator)
            logging.info(f'Saving at {self.config.log_dir}/sans_scanpath-latest_val_{ind}_{im_n}.png')
            image.save(f'{self.config.log_dir}/sans_scanpath-latest_val_{step_n}_{ind}_{im_n}.png')
            image = scanpath_over_pil_image(scanpath[0], image)
            image.save(f'{self.config.log_dir}/latest_val_{step_n}_{ind}_{im_n}.png')

    
    @torch.no_grad()
    def do_quant_val(self, val_dataloader, max_val_steps, dtype):
        logging.info(f'\nRunning validation for max {max_val_steps}\n')
        # fork_rng temporarily isolates changes
        with torch.random.fork_rng():
            # so you can change the seed here locally
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

def add_lora(transformer, rank, target_modules):
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank, 
        init_lora_weights="gaussian",
        target_modules=target_modules,
        )
    transformer.add_adapter(transformer_lora_config)
    logging.info(f"""trainable params: 
                 {transformer.num_parameters(only_trainable=True)} 
                 || all params: {transformer.num_parameters()}""")

@torch.no_grad()
def get_model_and_tokenizer(path, device, dtype, seed, do_compile, config):
    transformer = Flux2Transformer2DModel.from_pretrained("black-forest-labs/FLUX.2-klein-4B" if path is None
                                                           else path, # we save without a subdir
                                                           subfolder=None if path else 'transformer',
                                                           quantization_config=BitsAndBytesConfig(load_in_8bit=True,) if config.quantize_model else None,
                                                           strict=False)
    target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",             # double-stream attention
                "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",  # double-stream cross/context attention
                "to_qkv_mlp_proj",                              # single-stream fused qkv+mlp-in
                "to_out.0",                                     # single-stream fused attn-out+mlp-out
            ]
    if config.lora_path:
        transformer.load_lora_adapter(f'{config.lora_path}',
                                      prefix=None,
                                      adapter_name='default',
                                      target_modules=target_modules
                                      )
        transformer.set_adapters('default', 1)
    elif config.lora_rank:
        # we need a new lora as we aren't loading one
        # inplace operation
        add_lora(transformer, config.lora_rank, target_modules)

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
        pipe.vae.decode = torch.compile(pipe.vae.decode)
        pipe.vae.encode = torch.compile(pipe.vae.encode)
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
        optimizer = bnb.optim.PagedAdamW8bit(params, lr=lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=1)
    return optimizer, scheduler
