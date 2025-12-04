import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # Input: 3 channels (RGB), Output: 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Input: 32 channels, Output: 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        # Image reduces to 8x8 after two max_pools (32x32 -> 16x16 -> 8x8)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10) # 10 Classes for CIFAR-10

    def forward(self, x):
        # Convolution 1 + BatchNorm + ReLU + MaxPool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2) 
        
        # Convolution 2 + BatchNorm + ReLU + MaxPool
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_device():
    """Returns the CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    """Saves the model weights to a .pt file."""
    print(f"Saving model weights to {path}...")
    torch.save(model.state_dict(), path)
    print("Save complete.")

def load_model(path, device=None):
    """Initializes the architecture and loads weights from disk."""
    if device is None:
        device = get_device()
        
    print(f"Loading model from {path}...")
    model = SimpleClassifier().to(device)
    
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval() # Set to eval mode by default after load
        print("Model loaded successfully.")
        return model
    else:
        raise FileNotFoundError(f"No model found at {path}")

def export_to_onnx(model, onnx_path):
    """Exports a PyTorch model instance to ONNX format."""
    device = next(model.parameters()).device
    model.eval()
    
    print(f"Exporting model to {onnx_path}...")
    
    # Create dummy input based on device (1 image, 3 channels, 32x32)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    
    # Export
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=13)
    
    print(f"Export complete: {onnx_path}")

def train_model(epochs=2):
    """
    Trains the model and returns the trained model instance.
    Uses Automatic Mixed Precision (AMP) for speed on RTX cards.
    """
    device = get_device()
    print(f"--- Starting Training on {device} ---")

    # Data Loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Download CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

    # Setup Components
    model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Scaler for FP16 training
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

    # Training Loop
    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # AMP Forward Pass
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # AMP Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(trainloader):.4f} | Val Acc: {val_acc:.2f}%")

    return model


def run_torch_inference(model, input_tensor):
    """Runs inference on a loaded model instance."""
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        
    return probabilities.cpu().numpy()