import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from network import CNN
from helpers import write_conv, write_pool, write_dense


def main() -> None:
    r"""
    Main function.
    """
    # device setup
    device: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device}")

    # training settings
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3

    # data transformer f: [0, 255] -> [0, 1]
    transform: transforms = transforms.Compose([transforms.ToTensor()])

    # load mnist
    data_root: str = os.path.join(os.path.dirname(__file__), "mnist_torch")
    train_set: datasets = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_set: datasets = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)

    # set loaders
    train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader: DataLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # setup model
    model: CNN = CNN().to(device)
    optim: torch.optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # train loop
    for epoch in range(epochs):
        model.train()
        total_loss: float = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optim.zero_grad()
            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_loss: float = total_loss / len(train_loader)
        print(f"\r[  {epoch}/{epochs} epochs  {avg_loss:.4g} loss  ]", end="")

    # model eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs: torch.Tesnor = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\n{correct / total:.8g} acc%")

    # get state
    param_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "c_cnn", "rawnetwork", "parameters")
    params = {
        name: param.detach().cpu().numpy().astype(np.float32)
        for name, param in model.state_dict().items()
    }
    # conv1
    write_conv(
        kernels=params["conv1.weight"],
        biases=params["conv1.bias"],
        stride=(1, 1),
        file=os.path.join(param_dir, "conv1.bin")
    )
    # pool1
    write_pool(dims=(2, 2), stride=(2, 2), file=os.path.join(param_dir, "pool1.bin"))
    # conv2
    write_conv(
        kernels=params["conv2.weight"],
        biases=params["conv2.bias"],
        stride=(1, 1),
        file=os.path.join(param_dir, "conv2.bin")
    )
    # pool2
    write_pool(dims=(2, 2), stride=(2, 2), file=os.path.join(param_dir, "pool2.bin"))
    # dense1
    write_dense(
        weights=params["dense1.weight"],
        biases=params["dense1.bias"],
        file=os.path.join(param_dir, "dense1.bin")
    )

    # write dict
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "parameters_torch", "params.pth"))

    return None


if __name__ == "__main__":
    main()
