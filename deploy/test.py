import pytest
from flask import Flask, request, jsonify
import torch
import os
from serve import SimpleNet, app, model, predict
class TestPredict:
    
    app = Flask(__name__)
    model = SimpleNet()
    model.load_state_dict(torch.load("model/model.pt"))

    def test_happy_path_predict(self, mocker):
        files = {'image': open('tshirt-test.png', 'rb')}

        output = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        mocker.patch.object(model, "eval")
        model.eval.return_value = output

        with app.test_client() as client:
            response = client.post("/predict", data=files)

        assert response.status_code == 200
        assert response.json == {"predicted_class": 5}

    def test_edge_case_no_image(self):
        with app.test_client() as client:
            response = client.post("/predict")

        assert response.status_code == 400
        assert response.json == {"error": "No image provided"}