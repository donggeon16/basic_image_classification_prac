import torch
from pathlib import Path
from src.utils.constants import BASE, DEVICE


RUNS_DIR = BASE / "runs"


def main():
    best_files = sorted(RUNS_DIR.rglob("best.pt"))

    if not best_files:
        print("best.pt 파일이 없습니다.")
        return

    for path in best_files:
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )

            val_acc = checkpoint["val_acc"]
            folder = path.parent.relative_to(RUNS_DIR)

            print(f"{str(folder):<45} | Val Accuracy: {val_acc * 100:.3f}%")

        except Exception as e:
            print(f"{path}: 로드 실패 ({e})")


if __name__ == "__main__":
    main()