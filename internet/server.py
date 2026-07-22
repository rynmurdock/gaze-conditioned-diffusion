import asyncio
import io
import torch
import json
import time
import websockets

from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

# TODO update
import sys
sys.path.append('/home/ryn_mote/Misc/eye_experiments/gaze-conditioned-diffusion/src/')

# from pipe_modded_klein import Flux2KleinPipeline
# from modded_klein import Flux2Transformer2DModel
from data import scanpath_over_pil_image
from model import add_lora
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel


HOST = "localhost"
PORT = 8765
# TODO don't hardcode these
LATENT = torch.randn(1, 24 * 48, 128).to('cuda')
K = 18

USE_CIRCLE = False

LAST_K_POINTS = []


def w(t):
    return 4*t*(1-t)

def coords_to_pil_out(x, y, w, h):
    # 1. Create a blank image (width, height) and background color
    image = Image.new("RGB", (int(w), int(h)), "white")
    draw = ImageDraw.Draw(image)
    # A perfect square bounding box guarantees a perfect circle
    bounding_box = [w-int(x*w+32), h-int(y*h+32), w-int(x*w-32), h-int(y*h-32)]
    draw.ellipse(bounding_box, fill="blue", outline="black", width=3)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

import math
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)


import matplotlib.pyplot as plt
import numpy as np
def plot_heatmap(arr, cmap='viridis', title=None, cbar_label=None, figsize=(6, 5)):
    """
    Plot a heatmap from a 2D numpy array (height x width).
    """
    arr = np.array(arr.sum(0).cpu())
    assert arr.ndim == 2, f"Expected 2D array, got shape {arr.shape}"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=cmap, aspect='auto')
    fig.colorbar(im, ax=ax, label=cbar_label)

    plt.tight_layout()
    plt.savefig('this_fig.png')

def sub_point_wise_gaussians(tensor, coords, sigma=2.0, amplitude=2):
    H, W = tensor.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=tensor.device, dtype=torch.float32),
        torch.arange(W, device=tensor.device, dtype=torch.float32),
        indexing='ij'
    )
    cy, cx = coords[:, 0].view(-1, 1, 1), coords[:, 1].view(-1, 1, 1)
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    gaussians = amplitude * torch.exp(-dist_sq / (2 * sigma ** 2))  # (N, H, W)
    combined = gaussians.sum(0)  # (H, W)
    plot_heatmap(gaussians)
    return tensor + combined


def prepare_latents_mask(coords):
    global LATENT
    global LAST_K_POINTS
    
    # we wipe out our previous points after a long saccade followed by sustained gaze
    if len(coords[0]) > 4:
        last_three_distances = [distance(coords[0][j], coords[0][j+1]) for j in range(len(coords[0])-1)[-3:]]

        if last_three_distances[0] > 100 and last_three_distances[1] < 50 and last_three_distances[2] < 50:
            LAST_K_POINTS = LAST_K_POINTS[-2:]
        print(f'{last_three_distances=}')

    coords = coords.to(LATENT.device, LATENT.dtype)
    # vae downsampling
    coords //= 16
    # xy to yx
    coords = torch.flip(coords, (-1,))[0]
    gaze_mask = sub_point_wise_gaussians(LATENT.new_zeros(384//16, 768//16), coords)
    return gaze_mask


@torch.no_grad()
def coords_to_klein_out(coords) -> bytes:
    global LATENT
    try:
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor([coords]).to(torch.bfloat16)
        gaze_mask = prepare_latents_mask(coords)

        cond_img = scanpath_over_pil_image(coords[0], 
                                           w=768, 
                                           h=384, 
                                           just_path=True)
        
        
        def diffedit_callback_fn(pipe, step, timestep, kwargs):
            print(f'{kwargs['latents'].shape=}')

            # we're getting the "current" latent's timestep
            if step < 2:
                latents = kwargs['latents']
                sigma = pipe.scheduler.timesteps[step+1] / 1000
                eps = torch.randn_like(LATENT)
                # our LATENT is x0
                latent_renoised = LATENT * (1-sigma) + eps * (sigma)
                # pack mask
                print(f'{gaze_mask.sum()=}')
                mask = gaze_mask.reshape(1, gaze_mask.shape[-2] * gaze_mask.shape[-1], 1).expand(-1, -1, latents.shape[-1])
                mask = mask > (4-step)
                put_back = latent_renoised[mask]
                latents[mask] = put_back.to(latents.dtype)
                kwargs['latents'] = latents
            return kwargs

        image = pipe(
            image=cond_img,
            prompt='',
            height=384,
            width=768,
            guidance_scale=1.0,
            num_inference_steps=4,
            callback_on_step_end=diffedit_callback_fn,
            output_type='latent',
        ).images[0]
        LATENT = pipe._pack_latents(pipe._patchify_latents(image[None]))
        print(f'{LATENT.shape=}')
        image = pipe.vae.decode(image[None], return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type='pil')[0]
        image = scanpath_over_pil_image(torch.flip(coords[0], (0,),), image,
                                        color=(255,0,0,100))

    except Exception as e:
        print(e)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

if not USE_CIRCLE:
    device = "cuda"
    dtype = torch.bfloat16
    # TODO connect our config file here

    transformer = Flux2Transformer2DModel.from_pretrained('black-forest-labs/FLUX.2-klein-4B',
                                                          subfolder='transformer')
    transformer.load_lora_adapter('last_epoch_ckpt/pytorch_lora_weights.safetensors',
                                  prefix=None,
                                  use_safetensors=True)
    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
                                              transformer=transformer,
                                              torch_dtype=dtype)
    pipe = pipe.to(device, dtype)

    Image.open(io.BytesIO(coords_to_klein_out([[5, 7], [8, 10], [9, 9]]))).save('this.png')
    Image.open(io.BytesIO(coords_to_klein_out([[5, 7], [32, 123], [32, 111]]))).save('this.png')
    Image.open(io.BytesIO(coords_to_klein_out([[5, 7], [32, 123], [32, 111]]))).save('this.png')

    pipe.transformer = torch.compile(pipe.transformer)
    pipe.vae = torch.compile(pipe.vae)

# Single worker thread: GPU calls run here so they never block the
# asyncio event loop, and we never run two inference calls at once.
executor = ThreadPoolExecutor(max_workers=1)

def process_gaze(x: float, y: float, width: float, height: float, t: float) -> bytes:
    global LAST_K_POINTS

    x1, y1 = x * 764, y * 384
    LAST_K_POINTS.append((x1, y1))
    if len(LAST_K_POINTS) > K:
        LAST_K_POINTS.pop(0)

    try:
        """Runs on the executor thread. Returns raw PNG bytes."""
        if USE_CIRCLE:
            return coords_to_pil_out(x, y, width, height)
        else:
            return coords_to_klein_out(LAST_K_POINTS)
    except Exception as e:
        raise(e)

async def handler(websocket):
    print("Client connected.")
    loop = asyncio.get_running_loop()

    latest = None            # most recent (x, y, w, h, t), overwritten in place
    new_sample = asyncio.Event()

    async def receiver():
        nonlocal latest
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            keys = ["x", "y", "width", "height"]
            if any(data.get(k) is None for k in keys):
                continue

            x = float(data["x"]) / float(data["width"])
            y = float(data["y"]) / float(data["height"])
            width = float(data["width"])
            height = float(data["height"])
            client_t = data.get("t")

            # Overwrite, don't enqueue -- only the newest sample matters.
            latest = (x, y, width, height, client_t if client_t is not None else time.time())
            new_sample.set()

    async def worker():
        nonlocal latest
        while True:
            await new_sample.wait()
            new_sample.clear()
            sample = latest  # snapshot; receiver may overwrite `latest` while we run

            png_bytes = await loop.run_in_executor(executor, process_gaze, *sample)

            await websocket.send(png_bytes)  # binary frame, no disk round trip

    recv_task = asyncio.create_task(receiver())
    work_task = asyncio.create_task(worker())
    try:
        await asyncio.wait(
            [recv_task, work_task], return_when=asyncio.FIRST_COMPLETED
        )
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"Error in connection handler: {e}")
    finally:
        recv_task.cancel()
        work_task.cancel()
        print("Client disconnected.")


async def main():
    async with websockets.serve(handler, HOST, PORT, max_size=None):
        print(f"Gaze server listening on ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())