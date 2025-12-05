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

# ==========================================
# 1. The Architecture (Heavy VGG Style)
# ==========================================
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        
        # --- Block 1: 64 Channels ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # --- Block 2: 128 Channels ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # --- Block 3: 256 Channels ---
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        # --- Block 4: 512 Channels (The Heavy Lifter) ---
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # --- Classifier Head ---
        # Image reduces: 32 -> 16 -> 8 -> 4 -> 2 (4 MaxPools)
        # Final size: 512 channels * 2 * 2 pixels = 2048 features
        self.fc1 = nn.Linear(512 * 2 * 2, 2048)
        self.dropout = nn.Dropout(0.5) # Dropout to prevent memorization
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, NUM_CLASSES) 

    def forward(self, x):
        # Block 1 + Pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2) # 32 -> 16
        
        # Block 2 + Pool
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2) # 16 -> 8
        
        # Block 3 + Pool
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2) # 8 -> 4

        # Block 4 + Pool
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2) # 4 -> 2
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    
    torch.backends.cudnn.benchmark = False
    
    # CIFAR-100
    print("Loading CIFAR-100...")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    
    print(f"Moving {len(trainset)} images directly to {device} VRAM...")
    
    class GPUAugmentation(nn.Module):
        def __init__(self):
            super().__init__()
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4)
            )
        def forward(self, x):
            return self.aug(x)
    
    def prepare_gpu_dataset(dataset):
        # Extract data (N, H, W, C) -> (N, C, H, W) and scale to 0-1
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float() / 255.0
        targets = torch.tensor(dataset.targets).long()
        
        # Move to GPU immediately
        data = data.to(device)
        targets = targets.to(device)
        
        # Apply Normalization on GPU (Much faster)
        # (0.5, 0.5, 0.5) normalization
        data = (data - 0.5) / 0.5
        
        return torch.utils.data.TensorDataset(data, targets)

    # move train and valset to gpu:
    trainset = prepare_gpu_dataset(trainset)
    valset = prepare_gpu_dataset(valset)

    trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=0, pin_memory=False) 
    valloader = DataLoader(valset, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)

    model = SimpleClassifier().to(device)
    
    gpu_augment = GPUAugmentation().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            with torch.no_grad():
                inputs = gpu_augment(inputs).contiguous()
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
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
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)