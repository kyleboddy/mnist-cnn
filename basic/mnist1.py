"""
Module implementing a Convolutional Neural Network (CNN) for the MNIST dataset using PyTorch.
Includes functions for training the model, visualizing filters, activations, 
and the model's architecture.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torchviz import make_dot

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    """
    A Convolutional Neural Network for classifying images from the MNIST dataset.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self._to_linear = None
        self._calculate_to_linear()
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def _calculate_to_linear(self):
        """
        Calculate the size of the flattened layer.
        """
        sample_data = torch.zeros([1, 1, 28, 28])
        self._to_linear = self._forward_conv(sample_data).view(-1).shape[0]

    def _forward_conv(self, x):
        """
        Forward pass through convolutional layers for size calculation.
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        return x

    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = self._forward_conv(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.25)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Create a single instance of CNN
global_model = CNN()
optimizer = torch.optim.Adam(global_model.parameters())
criterion = nn.NLLLoss()

# Retrieve a single batch of images for visualization
images, _ = next(iter(train_loader))

# Training loop
for epoch in range(2):
    for batch_images, labels in train_loader:
        outputs = global_model(batch_images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Testing the model
    CORRECT = 0
    TOTAL = 0
    with torch.no_grad():
        for test_images, labels in test_loader:
            outputs = global_model(test_images)
            _, predicted = torch.max(outputs.data, 1)
            TOTAL += labels.size(0)
            CORRECT += (predicted == labels).sum().item()

    print(f"Epoch: {epoch+1}/{2} | Accuracy: {100 * CORRECT / TOTAL:.4f}%")

def visualize_filters(layer, filename):
    """
    Visualize the filters in a given convolutional layer.
    :param layer: The convolutional layer whose filters are to be visualized.
    :param filename: The filename to save the visualization to.
    """
    filters = layer.weight.data
    for i in range(filters.shape[0]):
        plt.imshow(filters[i][0], cmap="gray")
        plt.title(f"Filter {i}")
        plt.savefig(f"{filename}_filter_{i}.png")
        plt.close()

def visualize_activations(model, layer, image, filename):
    """
    Visualize the activations of a given convolutional layer for a specific image.
    :param model: The CNN model.
    :param layer: The convolutional layer.
    :param image: The image to generate activations for.
    :param filename: The filename to save the visualization to.
    """
    with torch.no_grad():
        # Forward pass through the initial part of the model up to the specified layer
        x = image[0:1].float()
        for layer_name, module in model.named_children():
            x = module(x)
            if layer_name == layer:
                break

        # Visualize activations
        activations = x
        for i in range(activations.shape[1]):
            plt.imshow(activations[0, i].cpu(), cmap="gray")
            plt.title(f"Activation {i}")
            plt.savefig(f"{filename}_activation_{i}.png")
            plt.close()

def visualize_model(model, input_size, filename):
    """
    Generate a visualization of the model's architecture.
    :param model: The model to be visualized.
    :param input_size: The size of the input tensor.
    :param filename: The filename to save the visualization to.
    """
    dummy_input = torch.zeros(input_size)
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render(filename, format="png")
    return dot

# Visualizations
visualize_model(global_model, (1, 1, 28, 28), "cnn_architecture")
visualize_filters(global_model.conv1, "conv1_filters")
visualize_activations(global_model, 'conv1', images, "conv1_activations")
visualize_filters(global_model.conv2, "conv2_filters")
visualize_activations(global_model, 'conv2', images, "conv2_activations")
