import torch
from pathlib import Path
from src.utils.seed import get_rng_state, restore_rng_state

BASE = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(epoch, model, optimizer, val_loss, val_acc, save_dir, is_best = False, scheduler = None):
    epoch += 1
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "rng_state": get_rng_state()
    }
    path = BASE/"runs"/save_dir
    path.mkdir(parents = True, exist_ok = True)
    if is_best:
        torch.save(checkpoint, path/"best.pt")
    else: 
        torch.save(checkpoint, path/f"epoch{epoch:03d}.pt")

def load_checkpoint(model, optimizer, save_dir, file_name, map_location = DEVICE, weights_only = True, scheduler = None):
    path = BASE/"runs"/save_dir/file_name
    best_path = BASE / "runs" / save_dir / "best.pt"
    best_val_acc = 0.0
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        best_val_acc = best_checkpoint["val_acc"]
    
    checkpoint = torch.load(path, map_location = map_location, weights_only = weights_only)
    model.load_state_dict(checkpoint["model_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        restore_rng_state(rng_state)
    saved_epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    val_acc = checkpoint["val_acc"]

    return saved_epoch, val_loss, val_acc, best_val_acc