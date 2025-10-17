import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from network import CNN
from helpers import write_tensor, write_label

epochs: int = 1_000
CNN = CNN()

# 1. Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # mean and std of MNIST
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=False, transform=transform),
    batch_size=1000,
    shuffle=False
)

# import mnist here??? fuck
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN.parameters(), lr=0.01)


for epoch in range(1, epochs):
    # loop & shit
    ...
