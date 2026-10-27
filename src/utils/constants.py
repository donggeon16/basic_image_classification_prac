from pathlib import Path
import torch

BASE = Path(__file__).resolve().parents[2]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)