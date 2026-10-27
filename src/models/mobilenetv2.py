import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(in_channels*expand_ratio)
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ) if not expand_ratio == 1 else None
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.use_shortcut = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.expansion(x) if self.expansion is not None else x
        x = self.depthwise(x)
        x = self.projection(x)
        return x + identity if self.use_shortcut else x


class MobileNetv2(nn.Module):
    def __init__(self, num_classes = 100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.stage1 = InvertedResidual(32, 16, stride=1, expand_ratio=1)
        self.stage2 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6)
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6)
        )
        self.stage4 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6)
        )
        self.stage5 = nn.Sequential(
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6)
        )
        self.stage6 = nn.Sequential(
            InvertedResidual(96, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6)
        )
        self.stage7 = InvertedResidual(160, 320, stride=1, expand_ratio=6)
        self.last = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.last(x)
        return self.fc(self.flat(self.gap(x)))



def test_invertedresidual():
    ivr1 = InvertedResidual(24, 24, 1, 6).to(DEVICE)
    x = torch.randn(8, 24, 32, 32, device=DEVICE)
    y1 = ivr1(x)
    print(y1.shape)
    ivr2 = InvertedResidual(24, 32, 2, 6).to(DEVICE)
    y2 = ivr2(x)
    print(y2.shape)
    print("backward test")
    loss1 = y1.sum()
    loss1.backward()
    loss2 = y2.sum()
    loss2.backward()
    print(ivr1.depthwise[0].weight.grad is not None)
    print(ivr2.depthwise[0].weight.grad is not None)

def test_mobilenetv2():
    model = MobileNetv2(num_classes=100).to(DEVICE)
    x = torch.randn(8, 3, 32, 32, device=DEVICE)
    y = model(x)
    print(y.shape)
    loss = y.mean()
    loss.backward()
    print("Gradient:", model.stem[0].weight.grad is not None)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")


def main():
    # test InvertedResidual
    # test_invertedresidual()
    # test MobileNetV2:
    test_mobilenetv2()


if __name__ == "__main__":
    main()