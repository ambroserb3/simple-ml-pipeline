import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

def evaluate_model(test_set, model):
    """
    Evaluates a simple nn model on the FashionMNIST dataset
    """
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")

if __name__ == "__main__":
    test_set = torch.load("data/test_set.pt")
    model = SimpleNet()
    model.load_state_dict(torch.load("model/model.pt"))
    evaluate_model(test_set, model)
