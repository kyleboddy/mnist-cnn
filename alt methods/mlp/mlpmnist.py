import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

def save_sample_images(loader, model, device, filepath, num_images=10):
    dataiter = iter(loader)
    images, labels = next(dataiter)  # Corrected line
    images = images.reshape(-1, INPUT_SIZE).to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()

    fig = plt.figure(figsize=(10, 4))
    for idx in np.arange(num_images):
        ax = fig.add_subplot(2, num_images//2, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"{preds[idx]} ({labels[idx]})", color=("green" if preds[idx]==labels[idx] else "red"))
    plt.savefig(filepath)
    plt.close()

# Define the plot_confusion_matrix function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Set device
device = torch.device("cpu")

# Hyperparameters
INPUT_SIZE = 784  # 28x28 images flattened
HIDDEN_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 4
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# MNIST dataset
train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# MLP neural network
class NeuralNet(nn.Module):
    """A simple Multilayer Perceptron network."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize the model
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# List to store loss values
loss_values = []

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        # Flatten images to (BATCH_SIZE, INPUT_SIZE)
        images = images.reshape(-1, INPUT_SIZE).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_values.append(loss.item())  # Store the loss value

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Plotting the training loss
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.close()

# Save the model checkpoint
torch.save(model.state_dict(), 'mnist_mlp.pth')

# Testing the model
model.eval()  # set the model to evaluation mode
all_preds = []
all_labels = []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, INPUT_SIZE).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.4f}%')

# Save model summary
with open('model_summary.txt', 'w') as f:
    f.write(str(model))

# Plot and save a confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm, classes=[str(i) for i in range(10)])
plt.savefig('confusion_matrix.png')
plt.close()

# Save some sample images with predictions
save_sample_images(test_loader, model, device, 'sample_predictions.png')
