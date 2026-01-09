import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# CONFIGURATION
IMG_SIZE = 32   # Native CIFAR resolution
NUM_CLASSES = 100 # CIFAR-100

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # --- Main Path ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- Shortcut Path ---
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        out = F.relu(out)
        return out

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.in_channels = 64
        
        # --- Initial Entry Block ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # --- ResNet Stages ---
        # 64 Channels (32x32)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        # 128 Channels (16x16)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        # 256 Channels (8x8)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        # 512 Channels (4x4)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        
        # --- Classifier Head ---
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, NUM_CLASSES)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Conv
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual Blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global Average Pooling & Linear
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_loader(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # CIFAR-100
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving model weights to {path}...")
    torch.save(model.state_dict(), path)
    print("Save complete.")


def load_model(path, device=None):
    if device is None:
        device = get_device()
    print(f"Loading model from {path}...")
    model = SimpleClassifier().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"No model found at {path}")


def export_to_onnx(model, onnx_path):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    print(f"Exporting model to {onnx_path}...")
    
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=18)
    print(f"Export complete: {onnx_path}")

def train_model(epochs=5):
    device = get_device()
    print(f"--- Starting Training on {device} ---")

    # Data Augmentation (Crucial for larger models to prevent overfitting)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # CIFAR-100
    print("Loading CIFAR-100...")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True) 
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    print("Data loaded.")
    model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

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
        scheduler.step(val_acc)
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

def get_test_loader(batch_size=64):
    """Returns the CIFAR-10 Test Set."""
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)