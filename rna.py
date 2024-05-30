import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import ImageOps, ImageFilter
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
import random
import glob

# Verificando se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregando o conjunto de dados MNIST
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

def create_digital_algarisms_dataset():
    # Crie um diretório para salvar as imagens, se ainda não existir
    if not os.path.exists('digit_images'):
        os.makedirs('digit_images')

    # Lista de tamanhos de fonte para variar o tamanho dos dígitos
    font_sizes = [32, 64, 128]

    # Get list of all font files in the Windows Fonts directory
    font_dir = 'C:\\Windows\\Fonts'
    font_files = [f for f in os.listdir(font_dir) if f.endswith('.ttf') or f.endswith('.otf')]

    # Select multiple random font files
    num_fonts = 5  # Change this to the number of fonts you want
    selected_font_files = random.sample(font_files, num_fonts)
    selected_fonts = [os.path.join(font_dir, font_file) for font_file in selected_font_files]

    # Select a random font size between 10 and 100 for each font
    selected_font_sizes = [random.randint(10, 100) for _ in range(num_fonts)]

    for digit in range(10):
        for font_size in selected_font_sizes:
            for font_style in selected_fonts:
                # Crie uma nova imagem com fundo branco
                img = Image.new('RGB', (font_size, font_size), color=(255, 255, 255))

                # Obtenha um objeto de desenho para desenhar na imagem
                d = ImageDraw.Draw(img)

                # Obtenha a fonte
                font = ImageFont.truetype(font_style, font_size)

                # Calcule a largura e a altura do dígito
                w, h = d.im.size

                # Desenhe o dígito na imagem
                d.text(((font_size - w) / 2, (font_size - h) / 2), str(digit), fill=(0, 0, 0), font=font)

                # Salve a imagem com um nome de arquivo que pode ser analisado corretamente
                img.save(f'digit_images/{digit}_{font_size}_{os.path.basename(font_style)}.png')

# Definindo o modelo
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

model = Net().to(device)  # Movendo o modelo para a GPU

# Definindo o otimizador e a função de perda
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
criterion = nn.CrossEntropyLoss()

# Check if the model and optimizer files exist
if os.path.exists('model.pth') and os.path.exists('optimizer.pth'):
    # Load the model and optimizer
    model.load_state_dict(torch.load('model.pth'))
    optimizer.load_state_dict(torch.load('optimizer.pth'))
else:
    # Treinando o modelo
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # Movendo os dados e os alvos para a GPU
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Save the model and optimizer
    torch.save(model.state_dict(), 'model.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')

# Testando o modelo
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)  # Movendo os dados e os alvos para a GPU
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Test Accuracy: %2d%%' % (100 * correct / total))

# Function to load a handwritten digit image and predict its label
def predict(image_path):
    image = Image.open(image_path).convert('L')
    
    # Invert the colors: Make the background black and the digits white
    image = ImageOps.invert(image)
    
    # Thicken the lines
    image = image.filter(ImageFilter.MaxFilter(3))  # You can adjust the size of the filter to make the lines thicker
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std as MNIST
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

def train_until_correct(image_path, correct_label):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std as MNIST
    ])
    image = transform(image).unsqueeze(0).to(device)
    label = torch.tensor([correct_label]).to(device)
    
    model.train()  # Set the model to training mode
    while True:
        output = model(image)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        if predicted.item() == correct_label:
            break

    model.eval()  # Set the model back to evaluation mode

    # Save the model and optimizer
    torch.save(model.state_dict(), 'model.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')

image_path = 'teste2.png'  # Substitua pelo caminho da sua imagem
correct_label = 8  # Substitua pela etiqueta correta

def train_from_image():
    num_trials = 1000  # Número de vezes para testar a imagem
    num_correct = 0  # Número de vezes que a previsão estava correta
    for _ in range(num_trials):
        predicted_label = predict(image_path)
        if predicted_label == correct_label:
            num_correct += 1
        else:
            print(predicted_label)
            train_until_correct(image_path, correct_label)
        accuracy = (num_correct / num_trials) * 100
        print(accuracy)
        if accuracy >= 90:
            print(num_correct, num_trials)
            break

def train_from_image_dataset():
    num_trials = 1000  # Número de vezes para testar a imagem
    num_correct = 0  # Número de vezes que a previsão estava correta

    # Carregue todas as imagens do diretório 'digit_images'
    image_files = glob.glob('digit_images/*.png')

    for _ in range(num_trials):
        # Escolha uma imagem aleatória e seu rótulo correto
        image_path = random.choice(image_files)
        correct_label = int(os.path.basename(image_path).split('_')[0])

        predicted_label = predict(image_path)
        if predicted_label == correct_label:
            num_correct += 1
        else:
            print(predicted_label)
            train_until_correct(image_path, correct_label)
        accuracy = (num_correct / num_trials) * 100
        print(accuracy)
        if accuracy >= 90:
            print(num_correct, num_trials)
            break

    print('Precisão:', accuracy, '%')

create_digital_algarisms_dataset()
train_from_image_dataset()
train_from_image()
for _ in range(50):
    print(predict(image_path))
