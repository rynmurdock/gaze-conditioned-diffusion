'''
python scripts/plots.py
'''
import json
import pandas as pd
from eval import plot_scores

csv_path = '/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/remote_gaze_logs/prompt_used.csv'
df = pd.read_csv(csv_path, index_col=0).to_dict()
plot_scores(dict([(k, [v['min_lpips'], v['min_cmmd']]) for k, v in df.items()]), 
            attribute='''Teacher's prompt''',
            score_types=['Best LPIPS (lower is better)', 'Best CMMD (lower is better)'], save_path=f'./outdoor_scenes_teacher_prompt.png')


csv_path = '/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/remote_gaze_logs/just_inf.csv'
df = pd.read_csv(csv_path, index_col=0).to_dict()
plot_scores(dict([(k, [v['min_lpips'], v['min_cmmd']]) for k, v in df.items()]), 
            attribute='''Full or inference-only timesteps''',
            score_types=['Best LPIPS (lower is better)', 'Best CMMD (lower is better)'], save_path=f'./outdoor_scenes_inf_timesteps.png')


cfg_one_job_csv_path = '''/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/remote_gaze_logs/lr=0.0001_lora_rank=128_max_steps=1010_batch_size=32_activation_checkpointing=True_use_prompt=The scene._teacher_use_prompt=_just_inf_timesteps=False/1000_ckpt/pytorch_lora_weights.safetensors_/plots/scores.csv'''
cols = ["lpips", "dinoscore", "cmmd"]
df = pd.read_csv(cfg_one_job_csv_path, index_col=0, converters={c: json.loads for c in cols}).T.to_dict()
plot_scores(dict([(k, [v['lpips'][0], v['cmmd'][0]]) for k, v in df.items()]), 
            attribute='Classifier free guidance scale', score_types=['Best LPIPS (lower is better)', 'Best CMMD (lower is better)'], save_path=f'./cfg_outdoor_scenes_scores.png')


