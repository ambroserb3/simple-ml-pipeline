import torch
from torchvision import datasets, transforms
import os

def download_and_prepare_data():
    """
    Downloads FashionMNIST dataset and saves it to data/ directory
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    torch.save(train_set, "data/train_set.pt")
    torch.save(test_set, "data/test_set.pt")

if __name__ == "__main__":
    download_and_prepare_data()
