import torch
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(epoch, model, optimizer, val_loss, val_acc, save_dir, is_best = False):
    epoch += 1
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc
    }
    path = BASE/"runs"/save_dir
    path.mkdir(parents = True, exist_ok = True)
    if is_best:
        torch.save(checkpoint, path/"best.pt")
    else: 
        torch.save(checkpoint, path/f"epoch{epoch:03d}.pt")

def load_checkpoint(model, optimizer, save_dir, file_name, map_location = DEVICE, weights_only = True):
    best_path = BASE / "runs" / save_dir / "best.pt"
    best_val_acc = 0.0
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        best_val_acc = checkpoint["val_acc"]
    
    checkpoint = torch.load(BASE/"runs"/save_dir/file_name, map_location = map_location, weights_only = weights_only)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    saved_epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    val_acc = checkpoint["val_acc"]

    return saved_epoch, val_loss, val_acc, best_val_acc