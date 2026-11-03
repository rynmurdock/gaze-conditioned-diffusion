'''
Manually place gaze points

python asso_works/manually_place_points_gradio.py

'''

import torch
import sys
import os
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data import scanpath_over_pil_image
from model import get_model_and_tokenizer
import config

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

import logging
logging.basicConfig(level=logging.INFO)

CANVAS_SIZE = (768, 384)


def blank_canvas() -> Image.Image:
    return Image.new("RGB", CANVAS_SIZE, "white")


def call_model(points: list[list[int]]) -> tuple[Image.Image, Image.Image]:
    """Returns (raw_image, overlay_image)."""
    # reshape(-1, 2) keeps an empty list as shape (0, 2) instead of (0,),
    # so scanpath[:, :2] inside scanpath_over_pil_image still works.
    points = np.array(points, dtype=int).reshape(-1, 2)
    print(points)
    with torch.autocast('cuda'):
        raw = model.inference(scanpath=points, guidance_scale=5)
        overlay = scanpath_over_pil_image(points, raw)
    return raw, overlay


def pick(show_overlay: bool, raw, overlay):
    """Choose which generated version to display."""
    return overlay if show_overlay else raw


def draw_points(image: Image.Image, points: list[list[int]]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    r = 5
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill="red", outline="white")
    return img


def add_point(base_image, points, evt: gr.SelectData):
    if base_image is None:
        base_image = blank_canvas()
    points = points + [[evt.index[0], evt.index[1]]]
    return draw_points(base_image, points), points


def on_upload(image):
    # Fired by the dedicated upload box (not the canvas).
    if image is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    resized = image.convert("RGB").resize(CANVAS_SIZE)
    # A user-supplied image replaces any generated result, so drop raw/overlay.
    # outputs: base_image, points, img, raw_image, overlay_image
    return resized, [], resized, None, None


def empty_upload_box():
    return None


def submit(points, show_overlay):
    # Empty list is valid: generates with no scanpath conditioning points
    raw, overlay = call_model(points)
    shown = pick(show_overlay, raw, overlay)
    # outputs: img, base_image, points (cleared), last_points, raw_image, overlay_image
    return shown, shown, [], points, raw, overlay


def regenerate(last_points, new_seed, show_overlay):
    # None = never submitted; [] = submitted with no points (valid)
    if last_points is None:
        raise gr.Error("Nothing to re-generate yet. Submit first.")
    if new_seed:
        model.config.seed = random.randint(0, 2**31 - 1)
        logging.info(f"Re-generating with seed {model.config.seed}")
    raw, overlay = call_model(last_points)
    shown = pick(show_overlay, raw, overlay)
    # outputs: img, base_image, points (cleared), raw_image, overlay_image
    return shown, shown, [], raw, overlay


def toggle_overlay(show_overlay, raw, overlay, points):
    # Nothing generated yet (blank canvas / uploaded image): leave display alone
    if raw is None or overlay is None:
        return gr.update(), gr.update()
    shown = pick(show_overlay, raw, overlay)
    # Redraw any pending (unsubmitted) points on top of the chosen version
    # outputs: img, base_image
    return draw_points(shown, points), shown


def reset_points(base_image):
    """Remove placed points but keep the current underlying image."""
    if base_image is None:
        base_image = blank_canvas()
    return base_image, []


def clear_canvas():
    """Wipe everything back to a blank canvas (including remembered points/results)."""
    blank = blank_canvas()
    # outputs: img, base_image, points, last_points, raw_image, overlay_image
    return blank, blank, [], None, None, None


with gr.Blocks() as demo:
    # States start empty; the functions fall back to a blank canvas when needed.
    base_image = gr.State(None)      # current image to place points on
    points = gr.State([])            # points currently being placed
    last_points = gr.State(None)     # points from the last submit (None = never submitted)
    raw_image = gr.State(None)       # last generation, no scanpath overlay
    overlay_image = gr.State(None)   # last generation, with scanpath overlay

    # Canvas: static display; clicks still fire .select
    img = gr.Image(
        value=blank_canvas(),
        type="pil",
        label="Click to place points",
        interactive=False,
    )

    # Dedicated upload box (click, drag+drop, clipboard paste).
    # Emptied after each upload so it is always ready for the next one.
    upload_box = gr.Image(
        value=None,
        type="pil",
        label="Optional: upload / drop an image to place points on",
        sources=["upload", "clipboard"],
    )

    with gr.Row():
        show_overlay = gr.Checkbox(value=True, label="Show scanpath overlay")
        new_seed = gr.Checkbox(value=True, label="Use a new random seed on re-generate")
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        regen_btn = gr.Button("Re-generate (same points)")
        clear_pts_btn = gr.Button("Clear points")
        clear_canvas_btn = gr.Button("Clear canvas (blank)")

    clear_outputs = [img, base_image, points, last_points, raw_image, overlay_image]

    upload_box.upload(
        on_upload, inputs=upload_box,
        outputs=[base_image, points, img, raw_image, overlay_image],
    ).then(empty_upload_box, inputs=None, outputs=upload_box)
    img.select(add_point, inputs=[base_image, points], outputs=[img, points])
    submit_btn.click(submit, inputs=[points, show_overlay],
                     outputs=[img, base_image, points, last_points, raw_image, overlay_image])
    regen_btn.click(regenerate, inputs=[last_points, new_seed, show_overlay],
                    outputs=[img, base_image, points, raw_image, overlay_image])
    show_overlay.input(toggle_overlay,
                       inputs=[show_overlay, raw_image, overlay_image, points],
                       outputs=[img, base_image])
    clear_pts_btn.click(reset_points, inputs=base_image, outputs=[img, points])
    clear_canvas_btn.click(clear_canvas, inputs=None, outputs=clear_outputs)

config = config.main_config
model = get_model_and_tokenizer(config.transformer_model_path, config.device,
                                config.dtype, config.seed, config.do_compile, config)
model.config.log_dir = './'
model.config.seed = 11

demo.launch()