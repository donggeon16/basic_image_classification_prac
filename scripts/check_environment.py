import sys
import torch
import torchvision

def check():
    t = torch.randn(3, 3, device="cuda", requires_grad=True)
    print(t)
    tt = (t**2).sum()
    tt.backward()
    print(t.grad)
    x = torch.tensor([2.0], device="cuda", requires_grad=True)
    y = x ** 2
    y.backward()
    print(x.grad)


def main():
    print("sys.version: ", sys.version)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)
    print("cuda on: ", torch.cuda.is_available())
    print("device: ", torch.cuda.get_device_name(0))
    check()


if __name__ == "__main__":
    main()