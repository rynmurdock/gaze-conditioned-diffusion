
import torch
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import get_model_and_tokenizer
from config import Config
from data import get_dataloader, scanpath_over_pil_image
from utils.eval_utils import get_dinoscore, get_lpips

config = Config.from_json('/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/logs/exodromic_Gemaric_Ceratitoidea/config.json')
config.lora_path = '/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/logs/exodromic_Gemaric_Ceratitoidea/28000_ckpt/pytorch_lora_weights.safetensors'
model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                    config.dtype, config.seed, config.do_compile, config)
model.pipe.transformer = torch.compile(model.pipe.transformer)
model.pipe.vae = torch.compile(model.pipe.vae)
model.config.log_dir = './'


model.config.seed = 11
torch.manual_seed(model.config.seed)
__train_dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_split_ratio,
                                                 config.batch_size, config.num_workers, config.seed,
                                                 config.resolution, config.use_cached_distilled_latents)

total_scores = 0
scores = {}
for data_ind, sample in enumerate(val_dataloader):
    if data_ind > 16:
        break
    total_scores += 1

    scanpaths = sample['scanpaths'][0]
    gt_image = sample['pil_images'][0]
    for ind in [1, 1.5, 1.6, 1.7, 1.8, 1.9]:
        with torch.autocast('cuda'):
            pred_image = model.inference(guidance_scale=ind, scanpath=scanpaths)
            dinoscore = get_dinoscore(pred_image, gt_image)
            lpips = get_lpips(pred_image, gt_image)
        print(f'{ind}: {dinoscore=}, {lpips=}')

        if f'guidance_scale={ind}' in scores:
            scores[f'guidance_scale={ind}'][0] += lpips
            scores[f'guidance_scale={ind}'][1] += dinoscore
        else:
            scores[f'guidance_scale={ind}'] = [lpips, dinoscore]

        with_scanpath = scanpath_over_pil_image(scanpaths, pred_image,)
        with_scanpath.save(f'scratch/{ind}_with_scanpath_pred.png')
        
        pred_image.save(f'scratch/{ind}_pred.png')
        gt_image.save(f'scratch/{ind}_gt.png')

# average our scores
scores = {k: [v / total_scores for v in vals] for k, vals in scores.items()}
fig, ax = plt.subplots(layout='constrained')
res = ax.grouped_bar(scores, tick_labels=('lpips', 'dinoscore'), group_spacing=1)
for container in res.bar_containers:
    ax.bar_label(container, padding=3)
fig.savefig('scratch/plot.png')


# TODO:
#   do we benefit from cfg on image? (start small like 1.2 -- + i doubt it)
#   lpips, image clipscore

