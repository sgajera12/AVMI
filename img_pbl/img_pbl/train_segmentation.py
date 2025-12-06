import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
from ae_segmentation import SegmentationAE, SegmentationVAE

class Rellis3DDataset(Dataset):
    def __init__(self, root_dir, sequences=None, img_size=(512, 512)):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        
        # Rellis-3D has sequences: 00000, 00001, 00002, 00003, 00004
        # Default: train on first 3, val on 4th, test on 5th
        if sequences is None:
            sequences = ['00000', '00001', '00002']
        
        self.img_base = self.root_dir / 'image'
        self.label_base = self.root_dir / 'image_id'
        
        # Collect all images from specified sequences
        self.image_paths = []
        self.label_paths = []
        
        for seq in sequences:
            img_dir = self.img_base / seq / 'pylon_camera_node'
            label_dir = self.label_base / seq / 'pylon_camera_node_label_id'
            
            if not img_dir.exists():
                print(f"Warning: {img_dir} does not exist, skipping")
                continue
            
            imgs = sorted(img_dir.glob('*.jpg'))
            for img_path in imgs:
                label_path = label_dir / img_path.name.replace('.jpg', '.png')
                if label_path.exists():
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.img_base} for sequences {sequences}")
        
        print(f"Found {len(self.image_paths)} image-label pairs from sequences {sequences}")
        
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        
        # Apply transforms
        image = self.img_transform(image)
        
        # For masks, don't use ToTensor (it normalizes to [0,1])
        # Instead, manually resize and convert
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        
        # Rellis-3D uses 0-34 for classes (35 total), plus potentially 255 for void
        # Keep all valid classes, only map truly invalid values to ignore_index
        mask[mask > 34] = 255
        mask = torch.from_numpy(mask)
        
        return image, mask


def vae_loss(recon, target, mu, logvar, kl_weight=0.0001):
    """Combined reconstruction and KL divergence loss for VAE"""
    # Use ignore_index=255 to skip invalid labels
    recon_loss = nn.CrossEntropyLoss(ignore_index=255)(recon, target)
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Normalize by number of latent elements
    num_latent_elements = mu.numel()
    kl_loss = kl_loss / num_latent_elements
    
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


def train_ae(model, train_loader, val_loader, epochs=50, device='cuda'):
    """Train basic Autoencoder"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
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
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ae_model.pth')
            print(f'Saved best model with val loss: {val_loss:.4f}')


def train_vae(model, train_loader, val_loader, epochs=50, device='cuda', kl_weight=0.0001):
    """Train Variational Autoencoder"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_total, train_recon, train_kl = 0.0, 0.0, 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs, mu, logvar = model(images)
            
            total_loss, recon_loss, kl_loss = vae_loss(outputs, masks, mu, logvar, kl_weight)
            
            total_loss.backward()
            optimizer.step()
            
            train_total += total_loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
        
        # Validation
        model.eval()
        val_total, val_recon, val_kl = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs, mu, logvar = model(images)
                
                total_loss, recon_loss, kl_loss = vae_loss(outputs, masks, mu, logvar, kl_weight)
                
                val_total += total_loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
        
        train_total /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        val_total /= len(val_loader)
        
        scheduler.step(val_total)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train - Total: {train_total:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}')
        print(f'Val   - Total: {val_total:.4f}')
        
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(model.state_dict(), 'best_vae_model.pth')
            print(f'Saved best model with val loss: {val_total:.4f}')


if __name__ == '__main__':
    # Config
    ROOT_DIR = '/home/pinaka/dataset/rellis3d/Rellis-3D'
    BATCH_SIZE = 8
    EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 35  # Rellis-3D has 35 classes (0-34)
    
    # Choose model type
    USE_VAE = True  # Set to False for basic AE
    
    # Rellis-3D sequence splits (sequences: 00000, 00001, 00002, 00003, 00004)
    # Standard split: train on 0,1,2, val on 3, test on 4
    #to start this with 
    train_sequences = ['00000', '00001', '00002']
    val_sequences = ['00003']
    
    # Data
    train_dataset = Rellis3DDataset(ROOT_DIR, sequences=train_sequences)
    val_dataset = Rellis3DDataset(ROOT_DIR, sequences=val_sequences)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    if USE_VAE:
        model = SegmentationVAE(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
        print("Training Variational Autoencoder...")
        train_vae(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE)
    else:
        model = SegmentationAE(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
        print("Training Basic Autoencoder...")
        train_ae(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE)