import asyncio
import io
import torch
import json
import time
import websockets

from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw
from diffusers import Flux2KleinPipeline


HOST = "localhost"
PORT = 8765

USE_CIRCLE = True

if not USE_CIRCLE:
    device = "cuda"
    dtype = torch.bfloat16
    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
    pipe = pipe.to(device)
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


@torch.no_grad()
def coords_to_klein_out(x, y, w, h) -> bytes:
    abs_h = 16
    abs_w = 16

    ar = h / w
    h = int(abs_h)
    w = int(abs_w / ar)
    latents = torch.randn(1, 128, h, w).to(device, dtype)

    xi = min(max(int(w * x), 0), w - 1)
    yi = min(max(int(h * y), 0), h - 1)
    H, W = latents.shape[2], latents.shape[3]
    y0, y1 = max(yi - 4, 0), min(yi + 4, H)
    x0, x1 = max(xi - 4, 0), min(xi + 4, W)
    latents[:, :, y0:y1, x0:x1] = -2

    image = pipe(
        prompt='fractal',
        height=16 * h,
        width=16 * w,
        guidance_scale=1.0,
        num_inference_steps=2,
        latents=latents,
        generator=torch.Generator(device=device).manual_seed(9)
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def process_gaze(x: float, y: float, width: float, height: float, t: float) -> bytes:
    try:
        """Runs on the executor thread. Returns raw PNG bytes."""
        if USE_CIRCLE:
            return coords_to_pil_out(x, y, width, height)
        else:
            return coords_to_klein_out(x, y, width, height)
    except Exception as e:
        print(e)

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