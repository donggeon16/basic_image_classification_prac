from torchvision.datasets import CIFAR10
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def main():
    dataset = CIFAR10(
    root=BASE/"data/raw",
    train=True,
    download=True
    )
    print("CIFAR10 데이터셋 크기: ", len(dataset))
    image, label = dataset[0]
    print(f"첫 번째 이미지의 자료형과 크기: {type(image)}, {image.size}")
    print(f"첫 번째 이미지의 레이블 ID: {label}")
    print(f"전체 클래스 개수와 클래스 이름: {len(dataset.classes)}, {dataset.classes}")


if __name__ == "__main__":
    main()