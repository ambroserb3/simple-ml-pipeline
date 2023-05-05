from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.datastructures import CombinedMultiDict

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

app = Flask(__name__)
model = SimpleNet()
model.load_state_dict(torch.load("mnt/app/model/model.pt"))

@app.route("/predict", methods=["POST"])
def predict():
    print("Request headers: ", request.headers)
    image = request.files.get("image")

    if request.content_type == "application/x-www-form-urlencoded":
        form = CombinedMultiDict([request.form, request.files])
        image = form.get("image")
    
    if image is None:
        return jsonify({"error": "No image provided"}), 400

    image = Image.open(image.stream).convert("L")
    image = image.resize((28, 28))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        model.eval()
        output = model(image_tensor)
        _, predicted_class = torch.max(output.data, 1)

    return jsonify({"predicted_class": int(predicted_class)}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
