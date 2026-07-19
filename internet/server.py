import asyncio
import io
import torch
import json
import time
import math
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
SEED = 9

USE_CIRCLE = False

LAST_12_POINTS = []

if not USE_CIRCLE:
    device = "cuda"
    dtype = torch.bfloat16
    # TODO connect our config file here
    transformer = Flux2Transformer2DModel.from_pretrained('black-forest-labs/FLUX.2-klein-4B',
                                                          subfolder='transformer')
    # TODO we've changed how this works in train.py
    add_lora(transformer, 16)
    # the diffusers lora weight & adapter loading are fucked for safetensors 
    # so we do it ourselves here 
    from safetensors.torch import load_file
    a = load_file('last_epoch_ckpt/diffusion_pytorch_model-00001-of-00002.safetensors')
    b = load_file('last_epoch_ckpt/diffusion_pytorch_model-00002-of-00002.safetensors')
    transformer.load_state_dict(a, strict=False)
    transformer.load_state_dict(b, strict=False)

    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
                                              transformer=transformer,
                                              torch_dtype=dtype)
    pipe = pipe.to(device, dtype)

    pipe.transformer = torch.compile(pipe.transformer)
    pipe.vae = torch.compile(pipe.vae)

# Single worker thread: GPU calls run here so they never block the
# asyncio event loop, and we never run two inference calls at once.
executor = ThreadPoolExecutor(max_workers=1)

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


def distance(p1, p2):
    print(p1)
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)

@torch.no_grad()
def coords_to_klein_out(coords) -> bytes:
    global SEED
    coords = torch.tensor([coords]).to(torch.bfloat16)

    if distance(coords[0][-1], coords[0][-2]) > 128:
        SEED = SEED + torch.randint(-10, 10, (1,)).item()

    print(coords.shape)
    try:
        cond_img = scanpath_over_pil_image(coords[0], 
                                           w=768, 
                                           h=384, 
                                           just_path=True)
        image = pipe(
            image=cond_img,
            prompt='',
            height=384,
            width=768,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.Generator(device=device).manual_seed(SEED)
        ).images[0]
        image = scanpath_over_pil_image(coords[0], image)

    except Exception as e:
        print(e)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def process_gaze(x: float, y: float, width: float, height: float, t: float) -> bytes:
    global LAST_12_POINTS

    x1, y1 = x * 764, y * 384
    LAST_12_POINTS.append((x1, y1))
    if len(LAST_12_POINTS) > 12:
        LAST_12_POINTS.pop(0)

    try:
        """Runs on the executor thread. Returns raw PNG bytes."""
        if USE_CIRCLE:
            return coords_to_pil_out(x, y, width, height)
        else:
            return coords_to_klein_out(LAST_12_POINTS)
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