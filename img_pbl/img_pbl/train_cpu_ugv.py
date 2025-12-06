"""
CPU-OPTIMIZED TRAINING FOR UGV SEGMENTATION

This script is designed for CPU training with:
- Minimal image size for speed
- Aggressive data sampling
- Small model for CPU efficiency
- Focus on off-road terrain classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

# Use a lightweight segmentation model for CPU
from torchvision.models.segmentation import fcn_resnet50

class FastRellis3DDataset(Dataset):
    """Super fast dataset for CPU training"""
    def __init__(self, root_dir, sequences=['00000'], img_size=(128, 128), sample_every=10):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        
        self.img_base = self.root_dir / 'image'
        self.label_base = self.root_dir / 'image_id'
        
        self.image_paths = []
        self.label_paths = []
        
        for seq in sequences:
            img_dir = self.img_base / seq / 'pylon_camera_node'
            label_dir = self.label_base / seq / 'pylon_camera_node_label_id'
            
            if not img_dir.exists():
                continue
            
            imgs = sorted(img_dir.glob('*.jpg'))
            # Sample every Nth frame for speed
            imgs = imgs[::sample_every]
            
            for img_path in imgs:
                label_path = label_dir / img_path.name.replace('.jpg', '.png')
                if label_path.exists():
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)
        
        print(f"Found {len(self.image_paths)} images (sampled)")
        
        # Augmentations to handle your darker environment
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.label_paths[idx])
        
        image = self.img_transform(image)
        
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        mask[mask > 34] = 255
        mask = torch.from_numpy(mask)
        
        return image, mask


class LightweightSegmentation(nn.Module):
    """Lightweight model for CPU training"""
    def __init__(self, num_classes=35):
        super().__init__()
        # Use FCN with ResNet50 - lighter than DeepLab
        self.model = fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.model(x)['out']


def train_cpu_optimized(model, train_loader, val_loader, epochs=5, device='cpu'):
    """Fast training optimized for CPU"""
    # Freeze backbone initially for speed
    for param in model.model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    print(f"\nTraining on {device}")
    print(f"Backbone frozen for first {epochs} epochs")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} [{i}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}')
        
        torch.save(model.state_dict(), f'ugv_segmentation_epoch{epoch+1}.pth')
    
    print("\nDone! Model saved.")


if __name__ == '__main__':
    # AGGRESSIVE CPU SETTINGS
    ROOT_DIR = '/home/pinaka/dataset/rellis3d/Rellis-3D'
    
    IMG_SIZE = (128, 128)      # Small for CPU speed
    BATCH_SIZE = 2             # Small batch for CPU memory
    EPOCHS = 5                 # Just 5 epochs for quick results
    SAMPLE_EVERY = 10          # Use every 10th frame (10x less data)
    NUM_WORKERS = 0            # 0 for CPU
    
    # Use just one sequence for super fast training
    train_sequences = ['00000']
    val_sequences = ['00003']
    
    DEVICE = 'cpu'  # Forcing CPU since CUDA isn't working
    NUM_CLASSES = 35
    
    print("="*60)
    print("CPU-OPTIMIZED TRAINING FOR UGV")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sample every {SAMPLE_EVERY} frames")
    print(f"Epochs: {EPOCHS}")
    print(f"Estimated time: 2-4 hours")
    print("="*60)
    
    # Data
    train_dataset = FastRellis3DDataset(ROOT_DIR, sequences=train_sequences, 
                                         img_size=IMG_SIZE, sample_every=SAMPLE_EVERY)
    val_dataset = FastRellis3DDataset(ROOT_DIR, sequences=val_sequences, 
                                       img_size=IMG_SIZE, sample_every=SAMPLE_EVERY)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Expected ~{len(train_loader)*2}s per epoch\n")
    
    # Model - lightweight for CPU
    model = LightweightSegmentation(num_classes=NUM_CLASSES).to(DEVICE)
    
    train_cpu_optimized(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE)
