import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets

class SimpleNet(nn.Module):
    """
    Simple neural network with 2 hidden layers
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(train_set):
    """
    Trains a model on the FashionMNIST dataset    
    """
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for _ in range(5):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "model/model.pt")

if __name__ == "__main__":
    train_set = torch.load("mnt/app/data/train_set.pt")
    train_model(train_set)
