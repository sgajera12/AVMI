import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from ae_segmentation import SegmentationVAE

class FastRellis3DDataset(Dataset):
    """Faster dataset with aggressive optimizations"""
    def __init__(self, root_dir, sequences=None, img_size=(128, 128), sample_every=2):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        
        if sequences is None:
            sequences = ['00000', '00001', '00002']
        
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
            # Sample every N frames for speed
            imgs = imgs[::sample_every]
            
            for img_path in imgs:
                label_path = label_dir / img_path.name.replace('.jpg', '.png')
                if label_path.exists():
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)
        
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
        
        image = self.img_transform(image)
        
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        mask[mask > 34] = 255
        mask = torch.from_numpy(mask)
        
        return image, mask


def fast_train_vae(model, train_loader, val_loader, epochs=20, device='cuda'):
    """Fast training with mixed precision"""
    optimizer = optim.AdamW(model.parameters(), lr=2e-3)  # Higher LR for faster convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision for speed
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs, mu, logvar = model(images)
                
                recon_loss = nn.CrossEntropyLoss(ignore_index=255)(outputs, masks)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / mu.numel()
                
                total_loss = recon_loss + 0.0001 * kl_loss
            
            # Mixed precision backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_total += total_loss.item()
        
        scheduler.step()
        
        # Validation (every 5 epochs to save time)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_total = 0.0
            with torch.no_grad(), torch.cuda.amp.autocast():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs, mu, logvar = model(images)
                    
                    recon_loss = nn.CrossEntropyLoss(ignore_index=255)(outputs, masks)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = kl_loss / mu.numel()
                    
                    total_loss = recon_loss + 0.0001 * kl_loss
                    val_total += total_loss.item()
            
            val_total /= len(val_loader)
            
            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save(model.state_dict(), 'fast_vae_model.pth')
        
        train_total /= len(train_loader)
        print(f'Epoch {epoch+1}/{epochs} - Train: {train_total:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')


if __name__ == '__main__':
    # AGGRESSIVE SPEED OPTIMIZATIONS
    ROOT_DIR = '/home/pinaka/dataset/rellis3d/Rellis-3D'
    
    # Speed settings
    IMG_SIZE = (128, 128)      # 128x128 instead of 512x512 = 16x faster
    BATCH_SIZE = 16            # Larger batch = fewer iterations
    EPOCHS = 20                # Fewer epochs with higher LR
    SAMPLE_EVERY = 3           # Use every 3rd frame = 3x less data
    NUM_WORKERS = 8            # More parallel data loading
    
    # Train on only 1 sequence for super fast testing
    train_sequences = ['00000']     # Just one sequence
    val_sequences = ['00003']
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 35
    
    print("="*60)
    print("FAST TRAINING MODE")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sample every {SAMPLE_EVERY} frames")
    print(f"Sequences: {train_sequences}")
    print("="*60)
    
    # Data
    train_dataset = FastRellis3DDataset(ROOT_DIR, sequences=train_sequences, 
                                         img_size=IMG_SIZE, sample_every=SAMPLE_EVERY)
    val_dataset = FastRellis3DDataset(ROOT_DIR, sequences=val_sequences, 
                                       img_size=IMG_SIZE, sample_every=SAMPLE_EVERY)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Estimated time per epoch: ~{len(train_loader)*0.3:.1f}s")
    
    # Smaller model for speed
    model = SegmentationVAE(in_channels=3, num_classes=NUM_CLASSES, latent_dim=64).to(DEVICE)
    
    print("\nTraining...")
    fast_train_vae(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE)
    print("\nDone! Model saved to fast_vae_model.pth")
