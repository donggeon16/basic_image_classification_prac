from torchvision.datasets import CIFAR100
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE / "data" / "splits"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    dataset = CIFAR100(
    root=BASE/"data/raw",
    train=True,
    download=True
    )
    print("CIFAR100 데이터셋 크기: ", len(dataset))
    image, label = dataset[0]
    print(f"첫 번째 이미지의 자료형과 크기: {type(image)}, {image.size}")
    print(f"첫 번째 이미지의 레이블 ID: {label}")
    print(f"전체 클래스 개수와 클래스 이름: {len(dataset.classes)}, {dataset.classes}")

    """
    fig, axes = plt.subplots(4, 5, figsize = (12, 10))
    for i in range(20):
        image, label = dataset[i]
        ax = axes[i//5, i%5]
        ax.imshow(image)
        ax.set_title(dataset.classes[label])
        ax.axis("off")
    plt.show()
    """

    """
    클래스 비율을 유지하면서 Train/Validation 분할
    고정된 Random Seed 사용 (예: 42)
    Train과 Validation의 인덱스가 겹치지 않는지 확인
    분할된 인덱스를 data/splits/에 저장
    """

    labels = np.array(dataset.targets)
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.1,
        random_state=42,
        stratify=labels
    )
    overlap = np.intersect1d(train_idx, val_idx)
    print(f"train_idx 길이: {len(train_idx)}\nval_idx 길이: {len(val_idx)}\n두 인덱스의 교집합 크기: {len(overlap)}\nTrain과 Val의 클래스별 이미지 개수: {np.bincount(labels[train_idx])}, {np.bincount(labels[val_idx])}")

    np.save(SAVE_DIR/"cifar100_train_idx.npy", train_idx)
    np.save(SAVE_DIR/"cifar100_val_idx.npy", val_idx)

    loaded_train_idx = np.load(SAVE_DIR/"cifar100_train_idx.npy")
    loaded_val_idx = np.load(SAVE_DIR/"cifar100_val_idx.npy")
    print(np.array_equal(train_idx, loaded_train_idx) and np.array_equal(val_idx, loaded_val_idx))
    
    """
    저장해 둔 cifar100_train_idx.npy 불러오기
    CIFAR-100 원본 데이터 중 Train에 해당하는 이미지만 선택하기
    픽셀 값을 [0, 1] 범위로 변환하기
    RGB 채널별 Mean, Std 계산하기
    결과와 배열 Shape 출력하기
    """

    raw_x_train = dataset.data[loaded_train_idx]
    x_train = raw_x_train.astype(np.float32) / 255.0
    x_mean = x_train.mean(axis = (0, 1, 2))
    x_std = x_train.std(axis = (0, 1, 2))
    print(f"mean: {x_mean}, std: {x_std}, shape: {x_train.shape}")

    path = BASE / "data" / "stats"
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path/"cifar100_stats.npz", mean = x_mean, std = x_std)
    stats = np.load(path/"cifar100_stats.npz")
    mean = stats["mean"]
    std = stats["std"]
    print(np.array_equal(x_mean, mean) and np.array_equal(x_std, std))


if __name__ == "__main__":
    main()