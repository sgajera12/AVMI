"""
BEST OPTION: Use Pretrained DeepLabV3
- Trains in 10-15 minutes instead of hours
- Much better results
- Already learned features from ImageNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from pathlib import Path
from train_fast import FastRellis3DDataset

class PretrainedSegmentation(nn.Module):
    """DeepLabV3 with pretrained ResNet50 backbone"""
    def __init__(self, num_classes=35):
        super().__init__()
        # Load pretrained DeepLabV3
        self.model = deeplabv3_resnet50(pretrained=True)
        
        # Replace classifier for our number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.model(x)['out']


def train_pretrained(model, train_loader, val_loader, epochs=10, device='cuda'):
    """Train pretrained model - much faster convergence"""
    # Only train classifier, freeze backbone initially
    for param in model.model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    best_val_loss = float('inf')
    
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
            scheduler.step()
            
            train_loss += loss.item()
            
            if i % 20 == 0:
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'pretrained_segmentation.pth')
            print('Saved best model')
    
    # Fine-tune: unfreeze backbone for last few epochs
    if epochs > 5:
        print("\nFine-tuning backbone...")
        for param in model.model.backbone.parameters():
            param.requires_grad = True
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        for epoch in range(3):  # Just 3 epochs of fine-tuning
            model.train()
            train_loss = 0.0
            
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            print(f'Fine-tune Epoch {epoch+1}/3 - Loss: {train_loss:.4f}')
        
        torch.save(model.state_dict(), 'pretrained_segmentation_finetuned.pth')


if __name__ == '__main__':
    ROOT_DIR = '/home/pinaka/dataset/rellis3d/Rellis-3D'
    
    # Fast settings
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 16
    EPOCHS = 10  # Only 10 epochs needed with pretrained model
    SAMPLE_EVERY = 2
    
    train_sequences = ['00000', '00001']  # 2 sequences
    val_sequences = ['00003']
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 35
    
    print("="*60)
    print("PRETRAINED MODEL TRAINING (FASTEST & BEST)")
    print(f"Using DeepLabV3 with ResNet50 backbone")
    print(f"Expected time: 10-15 minutes for {EPOCHS} epochs")
    print("="*60)
    
    # Data
    train_dataset = FastRellis3DDataset(ROOT_DIR, sequences=train_sequences, 
                                         img_size=IMG_SIZE, sample_every=SAMPLE_EVERY)
    val_dataset = FastRellis3DDataset(ROOT_DIR, sequences=val_sequences, 
                                       img_size=IMG_SIZE, sample_every=SAMPLE_EVERY)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    
    # Model
    model = PretrainedSegmentation(num_classes=NUM_CLASSES).to(DEVICE)
    
    print("\nTraining...")
    train_pretrained(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE)
    print("\nDone! Model saved.")
