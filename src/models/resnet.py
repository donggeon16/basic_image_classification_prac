import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.first = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu
        )
        self.second = nn.Sequential(
            self.conv2,
            self.bn2
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.second(x1)
        x3 = x2 + self.shortcut(x)
        y = self.relu(x3)
        return y

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        )
        self.block2 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 1)
        )
        self.block3 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1)
        )
        self.block4 = nn.Sequential(
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(512, 10)
        
    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        y = self.linear(self.flat(self.gap(x5)))
        return y

def test_basicblock():
    print("identity shortcut test")
    block1 = BasicBlock(64, 64, 1).to(device)
    x = torch.randn(8, 64, 32, 32, device=device)
    y1 = block1(x)
    print(y1.shape)
    print("projection shortcut test")
    block2 = BasicBlock(64, 128, 2).to(device)
    y2 = block2(x)
    print(y2.shape)
    print("backward test")
    loss1 = y1.sum()
    loss1.backward()
    loss2 = y2.sum()
    loss2.backward()
    print(block1.conv1.weight.grad is not None)
    print(block2.conv1.weight.grad is not None)

def test_resnet18():
    model = ResNet18().to(device)
    x = torch.randn(8, 3, 32, 32, device=device)
    y = model(x)
    print(y.shape)
    loss = y.mean()
    loss.backward()
    print(model.stem[0].weight.grad is not None)
    print(model.stem[0].weight.grad.device)

    
def main():
    # test_basicblock()
    test_resnet18()


if __name__ == "__main__":
    main()