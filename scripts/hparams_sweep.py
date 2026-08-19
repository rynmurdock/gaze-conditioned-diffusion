'''
python scripts/hparams_sweep.py
'''

import os
import sys
import logging
import itertools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from config import Config
from train import main

sweep_grid = {
    "lr": [1e-4],
#    'lora_path': [None],
    "lora_rank": [128],
    "max_steps": [1_010],
    'batch_size': [32],
    'activation_checkpointing': [True],
    'use_prompt': ['The scene.'],
    'teacher_use_prompt': ['',],
    'just_inf_timesteps': [True,],
    'included_data_subsets': [('OutdoorNatural', )],
    'shift_timesteps_resolution': [True],
}

def grid_configs(grid: dict):
    keys, values = zip(*grid.items())
    for combo in itertools.product(*values):
        yield Config(**dict(zip(keys, combo)))

to_sweep = grid_configs(sweep_grid)
for cfg in to_sweep:
    logging.info([f"Running: {n}={getattr(cfg, n)}" for n in sweep_grid.keys()])
    cfg.exp_name = "_".join([f"{n}={getattr(cfg, n)}" for n in sweep_grid.keys()])
    try:
        main(cfg)
    except Exception as e:
        logging.warning(f'{cfg.exp_name} failed with {e}.')


# ./logs/nonfuturity_Danice_Efland/: sample_teacher=False; 
