'''
python scripts/eval.py
'''

import torch
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import get_model_and_tokenizer
from config import Config
from data import get_dataloader, scanpath_over_pil_image
from utils.eval_utils import get_dinoscore, get_lpips



import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def plot_scores(scores, score_types, save_path='scratch/plot.png'):
    """
    Grouped bar chart of scores by guidance scale.
    """
    labels = tuple(score_types)
    groups = list(scores.keys())
    n_groups = len(groups)

    x = np.arange(len(labels))
    width = 0.85 / n_groups
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * n_groups), 5), layout='constrained')
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_groups))

    for i, group in enumerate(groups):
        vals = [scores[group][st] for st in range(len(labels))]
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=str(group), color=colors[i],
               edgecolor='white', linewidth=0.6)

    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Score Comparison Across Guidance Scales', fontsize=13,
                  fontweight='bold', pad=14)
    max_val = max(v for inner in scores.values() for v in inner)
    ax.set_ylim(0, max_val * 1.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True, color='#e5e5e5', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', length=0)

    leg = ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                     ncol=n_groups, fontsize=9.5, title='Guidance scale',
                     title_fontsize=10, handlelength=1.2, columnspacing=1.2)
    leg.get_title().set_fontweight('bold')

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return fig

# strongest baseline prompt
# config.use_prompt = 'The scene has distintive focal areas'
os.makedirs('./scratch/', exist_ok=True)

path = '/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/logs/apostatising_Laennec_Phiona'
config = Config.from_json(f'{path}/config.json')
config.lora_path = f'{path}/53000_ckpt/pytorch_lora_weights.safetensors'

model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                    config.dtype, config.seed, config.do_compile, config)
model.pipe.transformer = torch.compile(model.pipe.transformer)
model.pipe.vae = torch.compile(model.pipe.vae)
model.config.log_dir = './'


model.config.seed = 11
torch.manual_seed(model.config.seed)
__train_dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_split_ratio,
                                                 config.batch_size, config.num_workers, config.seed,
                                                 config.resolution, )

total_scores = 0
lpips_scores = {}
dino_scores = {}
for data_ind, sample in enumerate(val_dataloader):
    if data_ind >= 16:
        break
    total_scores += 1

    scanpaths = sample['scanpaths'][0]
    gt_image = sample['pil_images'][0]
    for ind in [1, 2, 3, 4, 5]:
        with torch.autocast('cuda'):
            pred_image = model.inference(guidance_scale=ind, scanpath=scanpaths)
            dinoscore = get_dinoscore(pred_image, gt_image)
            lpips = get_lpips(pred_image, gt_image)
        print(f'{ind}: {lpips=}, {dinoscore=}')

        if f'guidance_scale={ind}' in lpips_scores:
            lpips_scores[f'guidance_scale={ind}'][0] += lpips
            dino_scores[f'guidance_scale={ind}'][0] += dinoscore
        else:
            lpips_scores[f'guidance_scale={ind}'] = [lpips]
            dino_scores[f'guidance_scale={ind}'] = [dinoscore]

        with_scanpath = scanpath_over_pil_image(scanpaths, pred_image,)
        with_scanpath.save(f'scratch/{ind}_with_scanpath_pred.png')
        
        pred_image.save(f'scratch/{ind}_pred.png')
        gt_image.save(f'scratch/{ind}_gt.png')

# average our scores
print(f'{total_scores=}')
lpips_scores = {k: [round(v / total_scores, 2) for v in vals] for k, vals in lpips_scores.items()}
dino_scores = {k: [round(v / total_scores, 2) for v in vals] for k, vals in dino_scores.items()}

# setup in such a way that we could add additional score types
#   but lpips+dino are inverted & different range, so leaving separate rn
plot_scores(lpips_scores, score_types=['lpips (lower is better)'], save_path='scratch/lpips_plot.png')
plot_scores(dino_scores, score_types=['DINO Score (higher is better)'], save_path='scratch/dinoscore_plot.png')


df = pd.DataFrame({
    'lpips': lpips_scores,
    'dinoscore': dino_scores,
}).to_csv('./scratch/scores.csv')