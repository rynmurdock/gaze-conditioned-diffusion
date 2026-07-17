

###########################################
'''
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py
'''
###########################################


import sys
import torch
torch.set_float32_matmul_precision('high')
import logging
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import get_dataloader
from model import get_model_and_tokenizer, get_optimizer_and_lr_sched, get_loss
from config import main_config


logging.basicConfig(level=logging.INFO)

def main(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = get_model_and_tokenizer(config.model_path, config.device, config.dtype, config.seed, config.do_compile, config)

    # trained_params = [p for n, p in model.pipe.transformer.named_parameters() if 'to_q' in n]
    # not_trained = [p for n, p in model.pipe.transformer.named_parameters() if not 'to_q' in n]
    optimizer, lr_sched = get_optimizer_and_lr_sched(model.pipe.transformer.parameters(), 
                                                     config.lr, config)
    # for p in not_trained:
    #     p.requires_grad = False
    
    dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_split_ratio,
                                                 config.batch_size, config.num_workers, config.seed,
                                                 config.resolution, config.use_distilled_latents)
    
    train_losses = []
    inner_train_losses = []
    validation_losses = []
    total_inds = 0

    for epoch in range(config.epochs):
        for ind, batch in tqdm(enumerate(iter(dataloader))):
            if total_inds > config.max_steps:
                logging.info('Saving our transformer & ending training')
                if config.lora_rank:
                    model.pipe.fuse_lora()
                model.pipe.transformer.save_pretrained(f'{config.save_path}/last_epoch_ckpt', )
                sys.exit()
            if batch is None or \
                            (config.use_distilled_latents and batch.get('latents', None) is None):
                logging.warning(f'Skipping batch! {batch}')
                continue

            image = batch['images']
            scanpaths = batch['scanpaths']
            
            scanpaths = scanpaths.to(config.device)
            image = image.to(config.device, config.dtype)

            if total_inds % config.freq == 0:
                # NOTE autocasting because our fp32 training model is also our val model
                with torch.autocast(enabled=True, device_type='cuda', dtype=config.dtype):
                    model.do_qual_val()
                val_loss = model.do_quant_val(val_dataloader, config.max_val_steps, config.dtype)
                logging.info(f'{val_loss=:.4f}')
                if total_inds // config.freq != 0:
                    validation_losses.append(val_loss)
                if len(inner_train_losses) > 0:
                    if total_inds // config.freq != 0:
                        train_losses.append(sum(inner_train_losses)/len(inner_train_losses))
                    inner_train_losses = []

                train_losses = train_losses
                plt.plot(train_losses)
                plt.plot(validation_losses)
                plt.savefig('latest_loss_curves.png')
                plt.clf()

            loss, loss_logging_dict = get_loss(model, image, scanpaths, 
                                               latents=batch.get('latents'),
                                               timesteps=batch.get('timesteps'),
                                               noise_pred=batch.get('noise_preds'),
                                               )
            if total_inds % config.freq == 0:
                mse_loss = loss_logging_dict.get('mse_loss')
                logging.info(
                    f'Train MSE: {mse_loss}, '
                    f'Weighted Total: {loss.item()}'
                )
            inner_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()

            total_inds += 1
            if total_inds % config.freq == 0:
                logging.info('Saving our transformer')
                model.pipe.transformer.save_pretrained(f'{config.save_path}/last_epoch_ckpt',)



if __name__ == '__main__':
    assert main_config.batch_size == 1, 'we"ll need batched RoPE for higher batch size'
    main(main_config)

