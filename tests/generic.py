import sys
import logging
import requests
import numpy as np
from PIL import Image

sys.path.append('/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/src/')

from utils.eval_utils import pil_to_n1_1_tensor, get_dinoscore

def test_dinoscore():
    # Get images from Figure 11
    urls = [
        'https://github.com/google/dreambooth/blob/main/dataset/rc_car/03.jpg?raw=true', # reference from Fig 11
        'https://github.com/google/dreambooth/blob/main/dataset/rc_car/02.jpg?raw=true'# Real Sample from Fig 11
    ]
    images = [Image.open(requests.get(url, stream=True).raw)for url in urls]
    metric = get_dinoscore(images)
    logging.info(f'''
    DINO Score
    Expected: 0.770
    Calculated: {metric:.3f}''')
    assert abs(metric - 0.770) < 0.001, (
                    f'Metric is {abs(metric - 0.770)} away from known good')

def test_pil_to_n1_1_tensor():
    # Pure white/black image — triggers boundary values (0 and 255)
    img = Image.fromarray(np.array([[0, 255]], dtype=np.uint8))
    tensor = pil_to_n1_1_tensor(img)
    logging.info(f'Range: [{tensor.min().item(), tensor.max().item()}]')
    assert tensor.min() == -1
    assert tensor.max() == 1

test_pil_to_n1_1_tensor()
test_dinoscore()
