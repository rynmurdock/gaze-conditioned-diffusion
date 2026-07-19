"""
Quick visual sanity check: plots ordered fixations on top of their
stimulus image so you can eyeball whether the coordinates actually
line up (as opposed to being in some other coordinate space, e.g. a
letterboxed screen canvas).

Run:
    python -m tests.data_smoke_tests
"""

import os
import random

import matplotlib.pyplot as plt
from matio import load_from_mat
from PIL import Image

from src.data import _iter_records, _scalar_str, get_dataloader

ROOT = "trainSet"
MAT_PATH = "trainSet/allFixData.mat"
STIM_DIR = os.path.join(ROOT, "Stimuli")
N_SAMPLES = 6
OUT_PATH = "scanpath_check.png"

def collect_from_dataloader():
    pairs = []
    dataloader, _ = get_dataloader('trainSet', .1, 1, 11, 
                                   seed=0, resolution=(512, 512), 
                                   use_cached_distilled_latents=False)
    for ind, d in enumerate(dataloader):
        d['pil_images'][0].save('pil_im.jpg')
        pairs.append((str(ind), d['pil_images'][0], '_', d['scanpaths'][0]))
        if ind > 5:
            return pairs
    return pairs

def collect_samples(n=N_SAMPLES, seed=0):
    mat = load_from_mat(MAT_PATH)
    all_data = mat["allData"]

    pairs = []
    for key in all_data.keys():
        img_path = os.path.join(STIM_DIR, key)
        if not os.path.isfile(img_path):
            continue
        for rec in _iter_records(all_data[key]):
            fix = rec["data"]
            if fix.ndim == 2 and fix.shape[1] >= 2 and fix.shape[0] > 0:
                pairs.append((key, img_path, _scalar_str(rec["name"]), fix))

    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def plot_samples(samples, out_path=OUT_PATH):
    n = len(samples)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (key, img, subj, fix) in zip(axes, samples):
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        ax.imshow(img)
        x, y = fix[:, 0], fix[:, 1]
        ax.plot(x, y, "-", color="cyan", linewidth=1, alpha=0.8)
        ax.scatter(x, y, c=range(len(x)), cmap="autumn", s=40, zorder=3)
        ax.scatter(x[0], y[0], facecolors="none", edgecolors="lime", s=150, linewidths=2)  # start
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(str(i), (xi, yi), fontsize=7, color="white")
        ax.set_title(f"{key}  |  subj={subj}  |  img={img.size}", fontsize=9)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path} ({n} samples). Green circle = first fixation, "
          f"numbers = fixation order, red->yellow = time progression.")


if __name__ == "__main__":
    samples = collect_samples()
    plot_samples(samples)

    samples_processed = collect_from_dataloader()
    plot_samples(samples_processed, out_path='scanpath_dataloader.png')
