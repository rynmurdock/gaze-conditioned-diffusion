'''
Manually place gaze points
python src/gradio.py
'''

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

import logging
logging.basicConfig(level=logging.INFO)


def call_model(points: list[list[int]]) -> Image.Image:
    points = np.array(points)
    with torch.autocast('cuda'):
        image = model.inference(scanpath=points)
    return image



def draw_points(image: Image.Image, points: list[list[int]]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    r = 5
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill="red", outline="white")
    return img


def add_point(base_image, points, evt: gr.SelectData):
    points = points + [[evt.index[0], evt.index[1]]]
    return draw_points(base_image, points), points


def on_upload(image):
    return image, [], image


def submit(points):
    if not points:
        raise gr.Error("Place at least one point first")
    result = call_model(points)
    return result, result, []


def reset(base_image):
    return base_image, []


with gr.Blocks() as demo:
    base_image = gr.State(None)  # current image to place points on
    points = gr.State([])

    img = gr.Image(type="pil", label="Click to place points")
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear points")

    img.upload(on_upload, inputs=img, outputs=[base_image, points, img])
    img.select(add_point, inputs=[base_image, points], outputs=[img, points])
    submit_btn.click(submit, inputs=points, outputs=[img, base_image, points])
    clear_btn.click(reset, inputs=base_image, outputs=[img, points])

import torch
from model import get_model_and_tokenizer
from config import main_config


config = main_config
model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                    config.dtype, config.seed, config.do_compile, config)
model.config.log_dir = './'
model.config.seed = 11

demo.launch()