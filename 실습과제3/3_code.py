import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# PoseLSTM Model Definition
class PoseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


# Sample Data for demonstration
input_size = 10
hidden_size = 128
num_layers = 2
output_length = 3  # Assuming output_length is 3
output_size = output_length * input_size

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model and move it to the appropriate device
model = PoseLSTM(input_size, hidden_size, num_layers, output_size).to(device)


# Dummy DataLoader for demonstration
class DummyDataset(Dataset):
    def __init__(self, num_samples, input_length, input_size):
        self.data = np.random.randn(num_samples, input_length, input_size)
        self.targets = np.random.randn(num_samples, 3, input_size)  # Assuming output_length is 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)


train_loader = DataLoader(DummyDataset(100, 10, input_size), batch_size=16, shuffle=True)
test_loader = DataLoader(DummyDataset(20, 10, input_size), batch_size=16, shuffle=False)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track Losses
train_losses = []

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.view(targets.size(0), -1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save training losses
np.savetxt('train_losses.csv', np.array(train_losses), delimiter=',')

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('learning_curve.png')
plt.close()

# Evaluation loop
model.eval()
test_loss = 0.0
all_outputs = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        targets = targets.view(targets.size(0), -1)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        all_outputs.append(outputs.view(-1, 3, input_size).cpu().numpy())  # Assuming output_length is 3
        all_targets.append(targets.view(-1, 3, input_size).cpu().numpy())  # Assuming output_length is 3

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Plot results
all_outputs = np.concatenate(all_outputs, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Save comparison plots with error metrics
for i in range(output_length):
    plt.figure(figsize=(10, 5))
    plt.plot(all_targets[0, i, :], label='Actual')
    plt.plot(all_outputs[0, i, :], label='Predicted')
    plt.title(f'Pose Prediction {i + 1}')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.legend()
    mse = np.mean((all_targets[0, i, :] - all_outputs[0, i, :]) ** 2)
    mae = np.mean(np.abs(all_targets[0, i, :] - all_outputs[0, i, :]))
    plt.text(0.5, 0.5, f'MSE: {mse:.4f}, MAE: {mae:.4f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.savefig(f'pose_prediction_{i + 1}.png')
    plt.close()


# Save distribution plot of depth data
def plot_depth_distribution(depth_data):
    plt.figure(figsize=(10, 5))
    plt.hist(depth_data.flatten(), bins=50, color='blue', alpha=0.7, label='Depth Data')
    plt.xlabel('Depth Value')
    plt.ylabel('Frequency')
    plt.title('Depth Data Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('depth_data_distribution.png')
    plt.close()

