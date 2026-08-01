
import os
import sys
import logging
import itertools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from config import Config
from train import main

sweep_grid = {
    "lr":                  [1e-4, 2e-4, 1e-5],
    "lora_rank":           [2, 8, 32, 128],
}

def grid_configs(grid: dict):
    keys, values = zip(*grid.items())
    for combo in itertools.product(*values):
        yield Config(**dict(zip(keys, combo)))

for cfg in grid_configs(sweep_grid):
    logging.info([f"Running: {n}={getattr(cfg, n)}" for n in sweep_grid.keys()])
    cfg.exp_name = "_".join([f"{n}={getattr(cfg, n)}" for n in sweep_grid.keys()])
    try:
        main(cfg)
    except:
        logging.warning(f'{cfg.exp_name} failed.')

