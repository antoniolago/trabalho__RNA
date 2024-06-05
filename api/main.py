import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify, send_file
import random
import glob
import io

app = Flask(__name__)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
criterion = nn.CrossEntropyLoss()

# Load the model and optimizer if available
if os.path.exists('model.pth') and os.path.exists('optimizer.pth'):
    model.load_state_dict(torch.load('model.pth'))
    optimizer.load_state_dict(torch.load('optimizer.pth'))
else:
    raise FileNotFoundError("Model and optimizer files not found. Please train the model first.")

def normalize_image(image_path):
    image = Image.open(image_path).convert('L')
    image = ImageOps.invert(image)
    image = image.point(lambda p: p > 164 and 255)
    image = image.filter(ImageFilter.MaxFilter(3))
    image = image.resize((28, 28))
    return image

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('L')
    normalized_image = normalize_image(image)
    predicted_label = predict(normalized_image)
    
    return jsonify({
        "predicted_label": predicted_label
    })

@app.route('/test_image', methods=['POST'])
def test_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('L')
    normalized_image = normalize_image(image)
    predicted_label = predict(normalized_image)
    
    return jsonify({
        "predicted_label": predicted_label
    })

@app.route('/normalized_image', methods=['POST'])
def get_normalized_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('L')
    normalized_image = normalize_image(image)

    img_io = io.BytesIO()
    normalized_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
