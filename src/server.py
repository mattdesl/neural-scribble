import asyncio
import websockets
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import warnings
from PIL import Image
import time
import random
import os
from pathlib import Path

SEED = 1337

# Python
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch (CPU)
torch.manual_seed(SEED)

# Make PyTorch deterministic (may reduce performance)
torch.use_deterministic_algorithms(True, warn_only=True)

# Optional: make hash-based ops deterministic
os.environ["PYTHONHASHSEED"] = str(SEED)

# MPS (Apple Silicon) notes:
# - MPS does not guarantee full determinism
# - This still fixes all *RNG-driven* ops (torch.rand, randn, randint, etc.)
if torch.backends.mps.is_available():
    torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1) DEVICE & MODEL SETUP (MPS or CPU)
# ─────────────────────────────────────────────────────────────────────────────

# Suppress QuickGELU warning
warnings.filterwarnings(
    "ignore",
    message="These pretrained weights were trained with QuickGELU activation.*"
)
# Suppress the MPS argmax int64 warning
warnings.filterwarnings(
    "ignore",
    message=".*MPS: no support for int64 for argmax_argmin_out.*"
)

if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS backend")
else:
    device = "cpu"
    print("MPS not available; using CPU")

img_size = 24  # in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put red-channel images are img_size×img_size
ACCEPT_RGBA_COLOR = True  # <<— set to True if client will send RGBA
IS_HALF = False

# Select the working dtype: fp16 only on MPS; keep CPU in fp32 for compatibility/perf
TARGET_DTYPE = torch.float16 if (IS_HALF and device == "mps") else torch.float32

# model_name = "ViT-B-32"
# pretrained = "laion2b_s34b_b79k"

# ('MobileCLIP2-S0', 'dfndr2b'), ('MobileCLIP2-S2', 'dfndr2b'), ('MobileCLIP2-S3', 'dfndr2b'), ('MobileCLIP2-S4', 'dfndr2b')

# model_name = "ViT-B-32"
# pretrained = "openai"
 
MODEL_IMG_SIZE = 224
# open_clip exposes these on the model in recent versions:
# MODEL_IMG_SIZE = model.visual.image_size if isinstance(model.visual.image_size, int) else model.visual.image_size[0]

# model_name = "hf-hub:timm/ViT-B-32-SigLIP2-256"
# model_name = "hf-hub:timm/ViT-B-32-SigLIP2-256"
# model_name = "hf-hub:timm/ViT-L-16-SigLIP-384"
 
model_name = "ViT-B-32-quickgelu"
pretrained = "openai" 

# model_name = "ViT-B-16-quickgelu"
# pretrained = "dfn2b"
# dfn2b
# metaclip_400m
# laion400m_e31
# laion400m_e32
 
#  ViT-L-14-quickgelu
# model_name = "ViT-B-16-quickgelu"
# pretrained = "openai"

torch.set_float32_matmul_precision("high")  # improves mixed-precision matmuls

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained, device=device
)

# model, preprocess = open_clip.create_model_from_pretrained(model_name)

# Move model to desired precision (fp16 only on MPS)
if TARGET_DTYPE is torch.float16:
    model = model.half()
    # model = model.to(dtype=TARGET_DTYPE)

model = model.to(device=device, dtype=TARGET_DTYPE)
model = model.to(memory_format=torch.channels_last)

model = model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

MAX_N = 32  # pick a sensible upper bound for your batch size
_send_buf_cpu = torch.empty(MAX_N, dtype=torch.float32, device='cpu')  # persistent

# ─────────────────────────────────────────────────────────────────────────────
# 2) GLOBAL TEXT EMBEDDING + LOCK
# ─────────────────────────────────────────────────────────────────────────────
text_embedding = None                  # (1, D) baseline single-prompt embedding
pooled_text_embedding = None           # (1, D) pooled (e.g., embed-mean) embedding
prompt_set_embeddings = None           # (K, D) normalized embeddings per template
default_prompt = "a butterfly"             # default
text_prompt = None
reference_image_embedding = None  # (1, D)

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATIONS (shift + noise) — applied AFTER upsample, BEFORE normalize
# ─────────────────────────────────────────────────────────────────────────────
AUG_SHIFT_P = 0.0
AUG_NOISE_P = 0.0
AUG_MR_P = 0

AUG_MR_TARGETS = (16,img_size)
# AUG_MR_TARGETS = (16,img_size) if img_size <= 24 else (16, 24, img_size)

MR_VIEWS = (img_size//4, img_size//2, img_size)  # try (32, 64, 128, 224) too
# Shift in *model pixels* (224×224). Keep modest.
AUG_SHIFT_MAX = 4  # pixels

# Noise in 0..1 space (before mean/std). Subtle.
AUG_NOISE_STD = 0.005

CFG_W = 0.0  # <-- adjust strength as you like
negative_prompts = [
    "an unrecognizable scribble",
    "a chaotic abstract scribble",
    "a random tangle of lines with no recognizable subject",
    "pure abstract line art with no identifiable object",
    "a noisy texture made of random pen strokes",
    "a photograph",
]
negative_embedding = None              # (1, D)

REFERENCE_IMAGE_PATH = "public/sailboat.jpg"  # path from CWD

drawing_style = ""
# Prompt templates (fixed comma + typo)
prompt_templates = [
    # "{obj}",
    
    "a blurry photo of a {obj}",
    "a blurry representation of a {obj}",
    "a blurry image of a {obj}",
    "out of focus photo of a {obj}",
    
    # "a drawing of {obj}",
    # "a sketch of {obj}",
    # "a painting of {obj}",
    
    # "a drawing of {obj} on a white background",
    # "a painting of {obj} on a white background",
    # "a sketch of {obj} on a white background",
    # "a scribbly sketch of {obj} on a white background",
    # "doodle sketch of {obj} on a white background",
    # "scribble sketch of {obj} on a white background",
    # "rough sketch of {obj} on a white background",
    # "a doodle of {obj} on a white background",
    # "a painting of {obj} on a white background",
    # "a picture of {obj} on a white background",
    
    # "a scribble sketch of {obj}",
    # "crayon drawing of {obj}",
    # "color pencil drawing of {obj}",
    # "painting of {obj}",
    
    # "the colors of {obj}",
    # "a color palette of {obj}",
    # "smooth drawing of {obj}",
    # "smooth blur of {obj}",
    # "{obj}",
    # "blurry image of {obj}",
    # "blurry photo of {obj}",
    # "bokeh photo of {obj}",
    # "blurry bokeh of {obj}",
    # "blurry {obj}",
    # "blurry photograph of {obj}",
    # "out of focus photo of {obj}",
    
    # doodles for show
    # "single line scribble drawing of {obj}",
    # "a single line drawing of {obj}",
    # "a sketch of {obj}",
    # "a doodle of {obj}",
    # "ink sketch of {obj}",
    # "pen sketch of {obj}",
    # "drawing of {obj}",
    # "line drawing of {obj}",
    # "single-line drawing of {obj}",
    
    # "a pen drawing of {obj}",
    # "ink drawing of {obj}",
    # "{obj}",
    # "a doodle of {obj}",
    # "a line drawing of {obj}",
    # "a scribble of {obj}",
    # "a single-line drawing of {obj}",
    # "a scribbly drawing of {obj}",
    # "a single continuous line drawing of {obj}",
]

# Pooling options:
#  - "embed_mean" (recommended default): mean of template embeddings, renormalized, then dot
#  - "softmax": log-sum-exp over per-template similarities with temperature TAU
#  - "topk_mean": mean over top-K per-image template sims
#  - "mean": mean over per-template sims (kept for parity)
#  - "max": max over per-template sims
#  - "single": compare to a single templated prompt (fast, for ablations)
#  - "image" use an image instead of text
#  - "template_max" best performing template
POOLING = "embed_mean" if len(prompt_templates) > 1 else "single"
print("POOLING MODE", POOLING)
TOPK = 3                 # used only if POOLING == "topk_mean"
TAU = 0.2               # temperature for softmax pooling (smaller → closer to max)

# if POOLING == 'single':
#     text_prompt = prompt_templates[0].format(obj=text_prompt)

_text_lock = asyncio.Lock()

from pathlib import Path

with torch.no_grad():
    _prompt_a = "a painting of a face" # upright
    _prompt_b = "a painting of a face"

    tokens = tokenizer([_prompt_a, _prompt_b]).to(device)
    text_embeds = model.encode_text(tokens)          # (2, D)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    text_embed_a  = text_embeds[0:1]           # (1, D)
    text_embed_b = text_embeds[1:2]           # (1, D)

def softmin2(a: torch.Tensor, b: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    # tau smaller -> closer to min, tau larger -> smoother
    x = torch.stack([a, b], dim=0)  # (2,N)
    return -tau * torch.logsumexp(-x / tau, dim=0)  # (N,)

@torch.no_grad()
def _encode_images_from_paths(paths: list[Path], batch_size: int = 64) -> torch.Tensor:
    """
    Returns normalized image embeddings (N, D) for a list of image file paths.
    Uses same preprocess path as server: RGB -> resize -> normalize(mean/std) -> encode_image.
    """
    embs = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i+batch_size]

        # Load -> RGB -> float in [0,1] -> (N,3,H,W)
        imgs = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            # Convert to tensor in [0,1] with shape (3,H,W) using PIL->numpy->torch
            arr = np.asarray(img, dtype=np.uint8).copy()
            t = torch.from_numpy(arr).to(device=device)  # (H,W,3) uint8
            t = t.permute(2, 0, 1).contiguous()          # (3,H,W)
            t = t.to(dtype=torch.float32).div_(255.0)    # (3,H,W) float
            imgs.append(t)

        img_batch = torch.stack(imgs, dim=0)  # (N,3,H,W)

        # Resize to model size (matches server)
        img_batch = F.interpolate(img_batch, size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE),
                                  mode="bilinear", align_corners=False)

        # Normalize (matches server's simple_mode)
        img_batch = (img_batch - mean) / std

        # Match your memory format / dtype choices
        img_batch = img_batch.to(memory_format=torch.channels_last)
        img_batch = img_batch.to(device=device, dtype=TARGET_DTYPE, memory_format=torch.channels_last)

        emb = model.encode_image(img_batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embs.append(emb)

    return torch.cat(embs, dim=0)  # (N,D)


@torch.no_grad()
def _compute_dual_fitness_for_embeds(
    img_embeds_upright: torch.Tensor,
    img_embeds_upside_down: torch.Tensor,
) -> torch.Tensor:
    """
    Fitness = min(
        sim(upright image, 'a sketch of a blue jay'),
        sim(upside-down image, 'a sketch of a sailboat')
    )

    Inputs:
      img_embeds_upright: (N, D), normalized
      img_embeds_upside_down: (N, D), normalized

    Returns:
      (N,) fitness tensor
    """
    global text_embed_a, text_embed_b
    assert img_embeds_upright.shape == img_embeds_upside_down.shape

    sim_a = (img_embeds_upright @ text_embed_a.T).squeeze(-1)
    sim_b = (img_embeds_upside_down @ text_embed_b.T).squeeze(-1)

    return torch.minimum(sim_a, sim_b)
    # return softmin2(sim_a, sim_b)

@torch.no_grad()
def _compute_fitness_for_embeds(img_embeds: torch.Tensor) -> torch.Tensor:
    """
    img_embeds: (N,D) normalized
    returns sims: (N,) using your current POOLING logic and current global embeddings.
    NOTE: this is synchronous; we assume you're calling it outside the websocket loop.
    """
    global prompt_set_embeddings, pooled_text_embedding, text_embedding, negative_embedding

    if POOLING == "image":
        if reference_image_embedding is None:
            raise RuntimeError("reference_image_embedding is None; failed to load reference image.")
        sims = (img_embeds @ reference_image_embedding.T).squeeze(-1)
        return sims

    # Use the same pieces your handler uses
    Pe = prompt_set_embeddings     # (K,D)
    tbar = pooled_text_embedding   # (1,D)
    te_single = text_embedding     # (1,D)
    neg_emb = negative_embedding   # (1,D)

    # Positive similarity
    if POOLING == "embed_mean":
        pos_sims = (img_embeds @ tbar.T).squeeze(-1)
    elif POOLING == "softmax":
        S = img_embeds @ Pe.T
        pos_sims = (S / TAU).logsumexp(dim=1)
    elif POOLING == "template_max":
        S = img_embeds @ Pe.T
        pos_sims = S.max(dim=1).values
    elif POOLING == "topk_mean":
        S = img_embeds @ Pe.T
        K = min(TOPK, S.shape[1])
        pos_sims = S.topk(k=K, dim=1).values.mean(dim=1)
    elif POOLING == "mean":
        S = img_embeds @ Pe.T
        pos_sims = S.mean(dim=1)
    elif POOLING == "max":
        S = img_embeds @ Pe.T
        pos_sims = S.max(dim=1).values
    elif POOLING == "single":
        pos_sims = (img_embeds @ te_single.T).squeeze(-1)
    else:
        pos_sims = (img_embeds @ tbar.T).squeeze(-1)

    # Optional CFG-style negative
    if neg_emb is not None and CFG_W != 0.0:
        neg_sims = (img_embeds @ neg_emb.T).squeeze(-1)
        sims = pos_sims - CFG_W * neg_sims
    else:
        sims = pos_sims

    return sims

def _translate_with_white_fill(img: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    img: (3, H, W) in [0,1], white background assumed ~1.
    dy>0 shifts down, dx>0 shifts right. Areas uncovered are filled with white (1).
    """
    _, H, W = img.shape
    out = img.new_full(img.shape, 1.0)  # fill white

    # Source and destination ranges
    y0_src = max(0, -dy)
    y1_src = min(H, H - dy) if dy >= 0 else H
    x0_src = max(0, -dx)
    x1_src = min(W, W - dx) if dx >= 0 else W

    y0_dst = max(0, dy)
    y1_dst = y0_dst + (y1_src - y0_src)
    x0_dst = max(0, dx)
    x1_dst = x0_dst + (x1_src - x0_src)

    # Guard against empty slices
    if (y1_src > y0_src) and (x1_src > x0_src):
        out[:, y0_dst:y1_dst, x0_dst:x1_dst] = img[:, y0_src:y1_src, x0_src:x1_src]
    return out

def _maybe_mr_downup_batch(img_batch: torch.Tensor) -> torch.Tensor:
    """
    Multi-resolution view augmentation (down -> up), applied ONCE per batch.

    img_batch: (N, 3, H, W) in [0,1]
    Uses:
      - AUG_MR_P: probability to apply per batch
      - AUG_MR_TARGETS: tuple of target sizes (e.g., (16,) or (16,24))
    """
    if AUG_MR_P <= 0.0:
        return img_batch

    if torch.rand((), device=img_batch.device) >= AUG_MR_P:
        return img_batch

    if not AUG_MR_TARGETS:
        return img_batch

    # Pick ONE target resolution for the whole batch (common random numbers)
    idx = int(torch.randint(0, len(AUG_MR_TARGETS), (1,), device=img_batch.device).item())
    r = int(AUG_MR_TARGETS[idx])

    N, C, H, W = img_batch.shape
    # safety clamp so we don't accidentally "upscale" in the downstep
    r = max(1, min(r, H, W))
    if r == H and r == W:
        return img_batch

    # Downsample then upsample back to original size
    img_batch = F.interpolate(img_batch, size=(r, r), mode="bilinear", align_corners=False)
    img_batch = F.interpolate(img_batch, size=(H, W), mode="bilinear", align_corners=False)
    return img_batch

def _maybe_augment_batch(img_batch: torch.Tensor) -> torch.Tensor:
    """
    img_batch: (N, 3, H, W) in [0,1], float dtype on device.
    Applies shift and/or noise with 50% chance each, but crucially:
      - decisions are made once per *batch*
      - the SAME shift (dx,dy) is applied to all images when active
      - the SAME noise field is added to all images when active
    Returns same shape/dtype/device.
    """
    N, C, H, W = img_batch.shape

    # Translation (50% per batch)
    if torch.rand((), device=img_batch.device) < AUG_SHIFT_P:
        dx = int(torch.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1, (1,), device=img_batch.device).item())
        dy = int(torch.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1, (1,), device=img_batch.device).item())
        # Apply same shift to all
        for i in range(N):
            img_batch[i] = _translate_with_white_fill(img_batch[i], dy=dy, dx=dx)

    # Pixel jitter noise (50% per batch)
    if torch.rand((), device=img_batch.device) < AUG_NOISE_P:
        # Same noise field for all images (broadcast over N)
        noise = torch.randn((1, C, H, W), device=img_batch.device, dtype=img_batch.dtype) * AUG_NOISE_STD
        img_batch = (img_batch + noise).clamp_(0.0, 1.0)

    return img_batch

def _mr_views(img_batch_224: torch.Tensor) -> torch.Tensor:
    # img_batch_224: (N,3,224,224) in [0,1] or normalized space depending on where you call it
    views = []
    for r in MR_VIEWS:
        if r == MODEL_IMG_SIZE:
            v = img_batch_224
        else:
            v = F.interpolate(img_batch_224, size=(r, r), mode="bilinear", align_corners=False)
            v = F.interpolate(v, size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE), mode="bilinear", align_corners=False)
        views.append(v)
    # (V,N,3,224,224)
    return torch.stack(views, dim=0)


def _make_prompt_set(user_prompt: str) -> list[str]:
    return [t.format(style=drawing_style, obj=user_prompt) for t in prompt_templates]

@torch.no_grad()
def _encode_reference_image(path: str) -> torch.Tensor:
    """
    Load an image from disk and return a normalized CLIP image embedding (1, D)
    on the correct device/dtype.
    """
    img = Image.open(path).convert("RGB")
    # Use the OpenCLIP preprocess for the reference image
    img_t = preprocess(img).unsqueeze(0)  # (1, 3, MODEL_IMG_SIZE, MODEL_IMG_SIZE)

    # Move to device / dtype / memory format consistent with runtime images
    img_t = img_t.to(device=device, dtype=TARGET_DTYPE, memory_format=torch.channels_last)

    emb = model.encode_image(img_t)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

@torch.no_grad()
def _encode_texts(prompts: list[str]) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # (K, D) on device

@torch.no_grad()
def _encode_text(prompt: str) -> torch.Tensor:
    tokens = tokenizer([prompt]).to(device)
    emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # (1, D) on device

async def set_text_embedding(prompt: str) -> str:
    """
    Set:
      - text_embedding: (1, D) for the raw user prompt
      - prompt_set_embeddings: (K, D) for the templated variants
      - pooled_text_embedding: (1, D) pooled (embed-mean) of the template set
    """
    global text_embedding, text_prompt, prompt_set_embeddings, pooled_text_embedding
    prompt = prompt.strip()
    if not prompt:
        return "Error: empty prompt."
    with torch.no_grad():
        # Use the template if we're in 'single' mode
        if POOLING == "single":
            single_prompt = prompt_templates[0].format(style=drawing_style, obj=prompt)
            base = _encode_text(single_prompt)      # (1, D)
            text_prompt = single_prompt
            print('prompt:', single_prompt)
        else:
            base = _encode_text(prompt)             # (1, D)
            text_prompt = prompt

        pset = _make_prompt_set(prompt)
        if POOLING != 'single':
            print('prompt set:', pset)
        Pe = _encode_texts(pset)                    # (K, D)
        tbar = Pe.mean(dim=0, keepdim=True)
        tbar = tbar / tbar.norm(dim=-1, keepdim=True)

    async with _text_lock:
        text_embedding = base
        prompt_set_embeddings = Pe
        pooled_text_embedding = tbar
    return f"OK: text prompt set to '{prompt}'."

asyncio.run(set_text_embedding(default_prompt))

# Initialize negative embedding once at startup
with torch.no_grad():
    neg_embs = _encode_texts(negative_prompts)        # (K_neg, D)
    neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)
    negative_embedding = neg_embs.mean(dim=0, keepdim=True)
    negative_embedding = negative_embedding / negative_embedding.norm(dim=-1, keepdim=True)

if POOLING == "image":
    try:
        reference_image_embedding = _encode_reference_image(REFERENCE_IMAGE_PATH)
        print(f"Loaded reference image embedding from '{REFERENCE_IMAGE_PATH}'")
    except Exception as e:
        print(f"ERROR loading reference image '{REFERENCE_IMAGE_PATH}': {e}")
        reference_image_embedding = None

# Initialize the default embedding once at startup
# with torch.no_grad():
#     text_embedding = _encode_text(text_prompt)
#     null_embs = torch.cat([_encode_text(p) for p in null_prompts], dim=0)
#     null_embs = null_embs / null_embs.norm(dim=-1, keepdim=True)
#     negative_embedding = _encode_text(negative_prompt)

#     vocab_list = []
#     for word in exploration_vocab:
#         prompt = f"sketch of a {word}"
#         tokens = tokenizer([prompt]).to(device)
#         emb = model.encode_text(tokens)
#         emb = emb / emb.norm(dim=-1, keepdim=True)
#         vocab_list.append(emb)
#     vocab_embeddings = torch.cat(vocab_list, dim=0)  # (V, D)

# Mean/Std in target dtype on device
# mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
#                     device=device, dtype=TARGET_DTYPE).view(1, 3, 1, 1)
# std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
#                     device=device, dtype=TARGET_DTYPE).view(1, 3, 1, 1)


# Prefer model-provided stats when available
try:
    mean = torch.tensor(model.visual.image_mean, device=device, dtype=torch.float32).view(1,3,1,1)
    std  = torch.tensor(model.visual.image_std,  device=device, dtype=torch.float32).view(1,3,1,1)
except Exception:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=torch.float32).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=torch.float32).view(1,3,1,1)


# ─────────────────────────────────────────────────────────────────────────────
# 3) WEBSOCKET HANDLER
# ─────────────────────────────────────────────────────────────────────────────
async def handler(websocket, path=None):
    global _send_buf_cpu, text_embedding, null_embs, negative_embedding, vocab_embeddings
    iteration = 0
    try:
        while True:
            message = await websocket.recv()

            # Branch: text frame → update the global text embedding
            if isinstance(message, str):
                status = await set_text_embedding(message)
                # await websocket.send(status)
                print(status)
                continue

            # Otherwise we expect a binary payload of uint8 pixels
            if not isinstance(message, (bytes, bytearray)):
                await websocket.send("Error: Expected binary payload of uint8 pixels or a text prompt.")
                continue

            arr = np.frombuffer(message, dtype=np.uint8).copy()
            # Zero-copy view of the incoming bytes; safe since we only read it once
            # arr = np.frombuffer(message, dtype=np.uint8)
            
            total = arr.size
            
            # Channels depend on the toggle
            channels = 4 if ACCEPT_RGBA_COLOR else 1
            per_img = img_size * img_size * channels

            if total % per_img != 0:
                await websocket.send(
                    f"Error: Payload length {total} not divisible by {per_img} "
                    f"(img_size={img_size}, channels={channels})."
                )
                continue

            N = total // per_img

            if N > _send_buf_cpu.numel():  # grow once if needed (still avoids per-call churn)
                _send_buf_cpu = torch.empty(N, dtype=_send_buf_cpu.dtype, device='cpu')

            with torch.inference_mode():
                start_time = time.time()

                if ACCEPT_RGBA_COLOR:
                    # Expect NHWC (RGBA interleaved) from client
                    rgba_np = arr.reshape((N, img_size, img_size, 4))                # (N, H, W, 4) uint8
                    rgba_cpu = torch.from_numpy(rgba_np)                              # CPU uint8
                    rgba_dev = rgba_cpu.to(device)                                    # device uint8
                    rgba_dev = rgba_dev.permute(0, 3, 1, 2).contiguous()              # (N, 4, H, W)
                    rgb = rgba_dev[:, :3, :, :].to(dtype=torch.float32).div_(255.0)   # drop alpha → (N, 3, H, W)
                else:
                    # Expect N×H×W uint8 red-channel images
                    red_images = arr.reshape((N, img_size, img_size))                 # (N, H, W)
                    
                    red_u8_cpu = torch.from_numpy(red_images)                          # CPU uint8
                    red_u8_dev = red_u8_cpu.to(device)
                    red = red_u8_dev.to(dtype=TARGET_DTYPE).unsqueeze(1)              # (N, 1, H, W)
                    red = red.div_(255.0)
                    rgb = red.expand(-1, 3, -1, -1)                       # (N, 3, H, W)

                # img_batch = F.interpolate(
                #     rgb, size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE), mode="bicubic", align_corners=False, antialias=True,
                # )
                
                # if the above doesn't work
                img_batch = F.interpolate(
                    rgb, size=(MODEL_IMG_SIZE, MODEL_IMG_SIZE), mode="bilinear", align_corners=False
                )
                
                img_batch = _maybe_mr_downup_batch(img_batch)
                
                simple_mode = True
                if simple_mode:
                    # img_batch = (img_batch - mean) / std
                    # img_batch = img_batch.to(memory_format=torch.channels_last)
                    # img_batch = img_batch.to(device=device, dtype=TARGET_DTYPE, memory_format=torch.channels_last)

                    # # Upright
                    # img_embeds_upright = model.encode_image(img_batch)
                    # img_embeds_upright = img_embeds_upright / img_embeds_upright.norm(dim=-1, keepdim=True)

                    # # Upside-down (180° rotation)
                    # img_batch_ud = torch.flip(img_batch, dims=(2, 3))
                    # img_embeds_upside_down = model.encode_image(img_batch_ud)
                    # img_embeds_upside_down = img_embeds_upside_down / img_embeds_upside_down.norm(dim=-1, keepdim=True)

                    # sims = _compute_dual_fitness_for_embeds(
                    #     img_embeds_upright,
                    #     img_embeds_upside_down,
                    # )
                    
                    img_batch = (img_batch - mean) / std
                    img_batch = img_batch.to(memory_format=torch.channels_last)
                    img_batch = img_batch.to(device=device, dtype=TARGET_DTYPE, memory_format=torch.channels_last)
                    img_embeds = model.encode_image(img_batch)
                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                    sims = _compute_fitness_for_embeds(img_embeds)
                else:
                    # create multi-res views BEFORE normalize
                    views = _mr_views(img_batch)  # (V,N,3,224,224)
                    
                    # normalize each view
                    views = (views - mean) / std
                    views = views.to(device=device, dtype=TARGET_DTYPE)
                    # views = views.to(device=device, dtype=TARGET_DTYPE, memory_format=torch.channels_last)

                    # flatten V and N for a single encode call
                    V, N, C, H, W = views.shape
                    flat = views.view(V*N, C, H, W)

                    emb = model.encode_image(flat)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    emb = emb.view(V, N, -1)

                    # average embeddings across views then renorm (or average sims; both work)
                    img_embeds = emb.mean(dim=0)
                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

                    sims = _compute_fitness_for_embeds(img_embeds)

                # 7) CPU copy once, zero-copy send
                # torch.mps.synchronize()
                _send_buf_cpu[:N].copy_(sims.to(dtype=_send_buf_cpu.dtype, device='cpu', non_blocking=False))
                mv  = memoryview(_send_buf_cpu[:N].numpy())
                end_time = time.time()
                await websocket.send(mv)

                max_fit = float(sims.max().item())
                print(f"Processed {N} images in {(end_time - start_time)*1e3:.2f} ms (prompt='{text_prompt}') - max fitness: {max_fit:.2f}")
                iteration += 1
                
    except websockets.exceptions.ConnectionClosedOK:
        return
    except Exception as e:
        try:
            await websocket.send(f"Server exception: {e}")
        finally:
            return

# ─────────────────────────────────────────────────────────────────────────────
# 4) SERVER BOOT
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    print("\nStarting CLIP WebSocket server on ws://localhost:8765 …")
    print(f"Default text prompt: '{text_prompt}'")
    print(f"Expecting {'RGBA' if ACCEPT_RGBA_COLOR else 'R-only'} payloads "
          f"with img_size={img_size}")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run indefinitely

def rank_and_prefix_rename_folder(folder,
                                  batch_size: int = 64,
                                  dry_run: bool = False) -> list[tuple[int, Path, float, Path]]:
    """
    1) Collect *.png in folder
    2) Compute fitness vs current global prompt embeddings (POOLING as configured)
    3) Sort descending by fitness
    4) Rename to '0000-' + original_name, '0001-' + ..., preserving the original name after the prefix.

    Returns a list of (rank, old_path, fitness, new_path)
    """
    folder = Path(folder)
    paths = sorted(folder.glob("*.png"))
    if not paths:
        return []

    # Encode + score
    img_embeds = _encode_images_from_paths(paths, batch_size=batch_size)
    sims = _compute_fitness_for_embeds(img_embeds)  # (N,)
    sims_cpu = sims.detach().to("cpu", dtype=torch.float32).numpy()

    # Sort indices by descending similarity
    order = np.argsort(-sims_cpu)

    # Build target names
    planned: list[tuple[int, Path, float, Path]] = []
    for rank, idx in enumerate(order):
        old_p = paths[int(idx)]
        score = float(sims_cpu[int(idx)])
        prefix = f"{rank:04d}-"
        new_name = prefix + old_p.name
        new_p = old_p.with_name(new_name)
        planned.append((rank, old_p, score, new_p))

    # Two-phase rename to avoid collisions:
    # old -> .__tmp__<uuid>-name, then tmp -> final
    if dry_run:
        return planned

    # phase 1
    tmp_paths = []
    for rank, old_p, score, new_p in planned:
        tmp_p = old_p.with_name(f".__tmp__{rank:04d}__{old_p.name}")
        if tmp_p.exists():
            raise FileExistsError(f"Temp path exists already: {tmp_p}")
        old_p.rename(tmp_p)
        tmp_paths.append((tmp_p, new_p))

    # phase 2
    for tmp_p, new_p in tmp_paths:
        if new_p.exists():
            raise FileExistsError(f"Target path exists already: {new_p}")
        tmp_p.rename(new_p)

    return planned


import re

_rank_prefix_re = re.compile(r"^\d{4}-")

def strip_existing_rank_prefix(folder, dry_run: bool = False) -> list[tuple[Path, Path]]:
    folder = Path(folder)
    paths = sorted(folder.glob("*.png"))
    changes = []
    for p in paths:
        if _rank_prefix_re.match(p.name):
            new_name = _rank_prefix_re.sub("", p.name, count=1)
            new_p = p.with_name(new_name)
            changes.append((p, new_p))

    if dry_run:
        return changes

    # two-phase rename to avoid collisions among stripped names
    tmp = []
    for i, (old_p, new_p) in enumerate(changes):
        tmp_p = old_p.with_name(f".__tmp_strip__{i:04d}__{old_p.name}")
        old_p.rename(tmp_p)
        tmp.append((tmp_p, new_p))
    for tmp_p, new_p in tmp:
        if new_p.exists():
            raise FileExistsError(f"Target exists: {new_p}")
        tmp_p.rename(new_p)

    return changes


if __name__ == "__main__":
    asyncio.run(main())

    # asyncio.run(set_text_embedding("a parrot"))
    # folder = "/Users/matt/Downloads/parrot-2-small"
    # strip_existing_rank_prefix(folder)  # optional
    # rank_and_prefix_rename_folder(folder, batch_size=64)
