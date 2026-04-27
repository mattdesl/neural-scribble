import asyncio
import websockets
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import warnings
import time
import random
import os

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["PYTHONHASHSEED"] = str(SEED)
if torch.backends.mps.is_available():
    torch.manual_seed(SEED)

warnings.filterwarnings(
    "ignore",
    message="These pretrained weights were trained with QuickGELU activation.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*MPS: no support for int64 for argmax_argmin_out.*",
)

if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS backend")
else:
    device = "cpu"
    print("MPS not available; using CPU")

img_size = 32
MODEL_IMG_SIZE = 224

model_name = "ViT-B-32-quickgelu"
pretrained = "openai"

torch.set_float32_matmul_precision("high")

model, _, _ = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained, device=device
)

model = model.to(device=device, memory_format=torch.channels_last)
model = model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

MAX_N = 32
_send_buf_cpu = torch.empty(MAX_N, dtype=torch.float32, device="cpu")

# Global text embedding state
text_embedding = None
pooled_text_embedding = None
prompt_set_embeddings = None
default_prompt = "a butterfly"
text_prompt = None

prompt_templates = [
    "a drawing of {obj}",
    "a sketch of {obj}",
    "a painting of {obj}",
]

POOLING = "embed_mean" if len(prompt_templates) > 1 else "single"
print(f"Pooling mode: {POOLING}")
TOPK = 3
TAU = 0.2

_text_lock = asyncio.Lock()


def _make_prompt_set(user_prompt: str) -> list[str]:
    return [t.format(obj=user_prompt) for t in prompt_templates]


@torch.no_grad()
def _encode_texts(prompts: list[str]) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    emb = model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)


@torch.no_grad()
def _encode_text(prompt: str) -> torch.Tensor:
    tokens = tokenizer([prompt]).to(device)
    emb = model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)


@torch.no_grad()
def _compute_fitness_for_embeds(img_embeds: torch.Tensor) -> torch.Tensor:
    Pe = prompt_set_embeddings
    tbar = pooled_text_embedding
    te_single = text_embedding

    if POOLING == "embed_mean":
        return (img_embeds @ tbar.T).squeeze(-1)
    elif POOLING == "softmax":
        return (img_embeds @ Pe.T / TAU).logsumexp(dim=1)
    elif POOLING == "template_max":
        return (img_embeds @ Pe.T).max(dim=1).values
    elif POOLING == "topk_mean":
        S = img_embeds @ Pe.T
        K = min(TOPK, S.shape[1])
        return S.topk(k=K, dim=1).values.mean(dim=1)
    elif POOLING == "mean":
        return (img_embeds @ Pe.T).mean(dim=1)
    elif POOLING == "max":
        return (img_embeds @ Pe.T).max(dim=1).values
    elif POOLING == "single":
        return (img_embeds @ te_single.T).squeeze(-1)
    else:
        return (img_embeds @ tbar.T).squeeze(-1)

async def set_text_embedding(prompt: str) -> str:
    global text_embedding, text_prompt, prompt_set_embeddings, pooled_text_embedding
    prompt = prompt.strip()
    if not prompt:
        return "Error: empty prompt."
    with torch.no_grad():
        if POOLING == "single":
            single_prompt = prompt_templates[0].format(obj=prompt)
            base = _encode_text(single_prompt)
            text_prompt = single_prompt
            print("prompt:", single_prompt)
        else:
            base = _encode_text(prompt)
            text_prompt = prompt

        pset = _make_prompt_set(prompt)
        if POOLING != "single":
            print("prompt set:", pset)
        Pe = _encode_texts(pset)
        tbar = Pe.mean(dim=0, keepdim=True)
        tbar = tbar / tbar.norm(dim=-1, keepdim=True)

    async with _text_lock:
        text_embedding = base
        prompt_set_embeddings = Pe
        pooled_text_embedding = tbar
    return f"OK: text prompt set to '{prompt}'."


asyncio.run(set_text_embedding(default_prompt))

try:
    mean = torch.tensor(
        model.visual.image_mean, device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        model.visual.image_std, device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
except Exception:
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073], device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711], device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)


async def handler(websocket, path=None):
    global _send_buf_cpu
    try:
        while True:
            message = await websocket.recv()

            if isinstance(message, str):
                status = await set_text_embedding(message)
                print(status)
                continue

            if not isinstance(message, (bytes, bytearray)):
                await websocket.send(
                    "Error: Expected binary payload of uint8 pixels or a text prompt."
                )
                continue

            arr = np.frombuffer(message, dtype=np.uint8).copy()
            total = arr.size
            channels = 4
            per_img = img_size * img_size * channels

            if total % per_img != 0:
                await websocket.send(
                    f"Error: Payload length {total} not divisible by {per_img} "
                    f"(img_size={img_size}, channels={channels})."
                )
                continue

            N = total // per_img

            if N > _send_buf_cpu.numel():
                _send_buf_cpu = torch.empty(
                    N, dtype=_send_buf_cpu.dtype, device="cpu"
                )

            with torch.inference_mode():
                start_time = time.time()

                rgba_np = arr.reshape((N, img_size, img_size, 4))
                rgba_dev = torch.from_numpy(rgba_np).to(device)
                rgba_dev = rgba_dev.permute(0, 3, 1, 2).contiguous()
                rgb = rgba_dev[:, :3, :, :].to(dtype=torch.float32).div_(255.0)

                img_batch = F.interpolate(
                    rgb,
                    size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                img_batch = (img_batch - mean) / std
                img_batch = img_batch.to(memory_format=torch.channels_last)

                img_embeds = model.encode_image(img_batch)
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                sims = _compute_fitness_for_embeds(img_embeds)

                _send_buf_cpu[:N].copy_(
                    sims.to(
                        dtype=_send_buf_cpu.dtype,
                        device="cpu",
                        non_blocking=False,
                    )
                )
                end_time = time.time()
                await websocket.send(memoryview(_send_buf_cpu[:N].numpy()))

                max_fit = float(sims.max().item())
                print(
                    f"Processed {N} images in {(end_time - start_time)*1e3:.2f} ms "
                    f"(prompt='{text_prompt}') - max fitness: {max_fit:.2f}"
                )

    except websockets.exceptions.ConnectionClosedOK:
        return
    except Exception as e:
        try:
            await websocket.send(f"Server exception: {e}")
        finally:
            return


async def main():
    print("\nStarting CLIP WebSocket server on ws://localhost:8765 …")
    print(f"Default text prompt: '{text_prompt}'")
    print(f"Expecting RGBA payloads with img_size={img_size}")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
