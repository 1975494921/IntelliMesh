import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import IntelliMesh.nn as nn
from IntelliMesh import Tensor
from IntelliMesh.activation import ReLu, Softmax
from IntelliMesh.loss_func import CrossEntropyLoss
from IntelliMesh.optimizer import Adam

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

out_features = 10
batch_size = 64
learning_rate = 0.001
epochs = 100

model = nn.Sequential([
    nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    ReLu,
    nn.Conv2D(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
    ReLu,
    nn.Conv2D(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1),
    nn.Flatten(),
    nn.Linear(49, 32),
    ReLu,
    nn.Linear(32, out_features),
])

optimizer = Adam(model, lr=learning_rate)
loss_function = CrossEntropyLoss


def test_model_accuracy(model, test_loader):
    N = len(test_loader.dataset)
    correct = 0
    for images, labels in test_loader:
        images = np.array(images)
        labels = np.array(labels)

        inputs = Tensor(images, requires_grad=False)
        outputs = model(inputs)
        predicted = np.argmax(outputs.data, axis=1)
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / N
    return accuracy


for epoch in range(epochs):
    loss_list = []

    for i, (images, labels) in enumerate(train_loader):
        images = np.array(images)
        labels = np.array(labels)

        inputs = Tensor(images, requires_grad=False)
        targets = Tensor(labels, requires_grad=False)

        # Forward pass
        predictions = model(inputs)

        loss = loss_function(predictions, targets)
        loss_list.append(float(loss.__str__()))

        if i % 20 == 0:
            print(f'Epoch {epoch}, batch {i}, loss: {loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = test_model_accuracy(model, test_loader)
    print(f'Epoch {epoch}, loss: {np.mean(loss_list if loss_list else [0])}, accuracy: {accuracy}%')
