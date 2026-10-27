import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.constants import BASE

class TrainingHistory:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def update(self, epoch, train_loss, val_loss, val_acc):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc * 100)

    def save_plot(self, path):
        
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Loss: 왼쪽 Y축
        line1, = ax1.plot(
            self.epochs, self.train_losses,
            label="Train Loss", color="blue"
        )
        line2, = ax1.plot(
            self.epochs, self.val_losses,
            label="Validation Loss", color="orange"
        )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax1.grid(True, alpha=0.3)

        # Accuracy: 오른쪽 Y축
        ax2 = ax1.twinx()

        line3, = ax2.plot(
            self.epochs, self.val_accuracies,
            label="Validation Accuracy",
            color="green"
        )

        ax2.set_ylabel("Validation Accuracy (%)")
        ax2.set_ylim(0, 100)

        # 범례 통합
        lines = [line1, line2, line3]
        ax1.legend(lines, [line.get_label() for line in lines])

        ax1.set_title(f"{path} - training_curves")
        fig.tight_layout()

        plot_path = (
            BASE / "runs" / path / "training_curves.png"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(plot_path, dpi=150)
        plt.close(fig)