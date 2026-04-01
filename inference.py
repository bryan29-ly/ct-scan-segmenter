"""
Inference utilities: model loading and single-slice segmentation.

Weights are downloaded from Hugging Face Hub on first run.
"""

import torch
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

from model import PlainConvUNet

# ── Preprocessing constants (from training) ──────────────────────────────────

P05 = 13.0
P995 = 213.0
MEAN = 94.301
STD = 48.477
NUM_CLASSES = 54

# ── Hugging Face config ──────────────────────────────────────────────────────
HF_REPO = "bryan29-ly/ct-scan-segmenter"
HF_FILENAME = "checkpoint_best.pth"

WEIGHTS_DIR = Path(__file__).parent / "weights"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model() -> tuple[PlainConvUNet, torch.device]:
    """Download weights (if needed) and return (model, device) ready for inference."""
    device = _get_device()

    WEIGHTS_DIR.mkdir(exist_ok=True)
    local_path = WEIGHTS_DIR / HF_FILENAME

    if not local_path.exists():
        print(f"Downloading weights from {HF_REPO}…")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            local_dir=str(WEIGHTS_DIR),
        )

    model = PlainConvUNet(in_channels=1, num_classes=NUM_CLASSES).to(device)

    checkpoint = torch.load(
        local_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def preprocess(image: np.ndarray) -> np.ndarray:
    """Clip and normalize a 256×256 uint8 CT slice."""
    img = image.astype(np.float32)
    img = np.clip(img, P05, P995)
    img = (img - MEAN) / STD
    return img


@torch.no_grad()
def segment_slice(model_and_device, image: np.ndarray) -> np.ndarray:
    """
    Run segmentation on a single 256×256 grayscale CT slice.

    Returns a (256, 256) uint8 mask where each pixel ∈ [0, NUM_CLASSES].
    0 = background, 1–54 = anatomical structures.
    """
    model, device = model_and_device

    img = preprocess(image)
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

    logits = model(tensor)                     # (1, 54, 256, 256)
    probs = torch.sigmoid(logits)              # per-class probabilities

    # Threshold then argmax: only assign a class if its prob > 0.5
    above_threshold = probs > 0.5              # (1, 54, 256, 256)
    has_any = above_threshold.any(dim=1)       # (1, 256, 256)

    # Among classes above threshold, pick the one with highest prob
    masked_probs = probs * above_threshold.float()
    argmax = masked_probs.argmax(dim=1) + 1    # shift: class indices 1–54

    # Where no class passes threshold → background (0)
    mask = torch.where(has_any, argmax, torch.zeros_like(argmax))

    return mask.squeeze(0).cpu().numpy().astype(np.uint8)
