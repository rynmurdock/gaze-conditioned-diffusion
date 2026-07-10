

###########################################
'''
python src/train.py
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
    optimizer, lr_sched = get_optimizer_and_lr_sched(list(model.pipe.transformer.parameters()), 
                                                     config.lr)
    dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_split_ratio,
                                                 config.batch_size, config.num_workers, config.seed)
    
    train_losses = []
    inner_train_losses = []
    validation_losses = []
    total_inds = 0

    for epoch in range(config.epochs):
        for ind, batch in tqdm(enumerate(iter(dataloader))):
            if total_inds > config.max_steps:
                model.pipe.transformer.save_pretrained(f'{config.save_path}/last_epoch_ckpt', 
                                                  from_pt=True)
                sys.exit()
            if batch is None:
                continue

            x0 = batch['images']
            scanpaths = batch['scanpaths']
            
            scanpaths = scanpaths.to(config.device)
            x0 = x0.to(config.device)
            x0 = x0.to(config.dtype)

            if total_inds % config.freq == 0:
                # NOTE autocasting because our fp32 training model is also our val model; only want calculations in half.
                with torch.autocast(enabled=True, device_type='cuda', dtype=config.dtype): 
                    # TODO
                    # model.do_qual_val([[Image.open(j) for j in examples]], k=config.k)

                    val_loss = model.do_quant_val(val_dataloader, config.max_val_steps)
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

            optimizer.zero_grad()
            loss, loss_logging_dict = get_loss(model, x0, scanpaths)
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

            total_inds += 1
            if total_inds % config.freq == 0:
                # TODO add loading from path
                model.pipe.transformer.save_pretrained(f'{config.save_path}/last_epoch_ckpt', from_pt=True)

if __name__ == '__main__':
    main(main_config)
