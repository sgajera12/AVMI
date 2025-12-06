import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from ae_segmentation import SegmentationAE, SegmentationVAE
from train_segmentation import Rellis3DDataset

def compute_iou(pred, target, num_classes):
    """Compute mean IoU across classes, ignoring label 255"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Filter out ignore_index pixels
    valid_mask = target != 255
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return np.nanmean(ious)


def evaluate_model(model, test_loader, device='cuda', num_classes=20, is_vae=False):
    """Evaluate segmentation model"""
    model.eval()
    total_iou = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            if is_vae:
                outputs, _, _ = model(images)
            else:
                outputs = model(images)
            
            preds = torch.argmax(outputs, dim=1)
            
            # Filter out ignore_index for metrics
            valid_mask = masks != 255
            
            # Metrics
            iou = compute_iou(preds, masks, num_classes)
            acc = (preds[valid_mask] == masks[valid_mask]).float().mean().item()
            
            total_iou += iou
            total_acc += acc
    
    mean_iou = total_iou / len(test_loader)
    mean_acc = total_acc / len(test_loader)
    
    return mean_iou, mean_acc


def visualize_predictions(model, dataset, device='cuda', num_samples=4, is_vae=False):
    """Visualize model predictions"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i]
            image_input = image.unsqueeze(0).to(device)
            
            if is_vae:
                output, _, _ = model(image_input)
            else:
                output = model(image_input)
            
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Denormalize image
            img_display = image.cpu().numpy().transpose(1, 2, 0)
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.cpu().numpy(), cmap='tab20')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='tab20')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to segmentation_results.png")


if __name__ == '__main__':
    ROOT_DIR = '/home/pinaka/dataset/rellis3d/Rellis-3D'  # Update this
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 35  # Rellis-3D has 35 classes (0-34)
    
    # Choose model
    USE_VAE = True
    MODEL_PATH = 'best_vae_model.pth' if USE_VAE else 'best_ae_model.pth'
    
    # Load model
    if USE_VAE:
        model = SegmentationVAE(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    else:
        model = SegmentationAE(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Loaded model from {MODEL_PATH}")
    
    # Test data - sequence 00004 for testing
    test_dataset = Rellis3DDataset(ROOT_DIR, sequences=['00004'])
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Evaluate
    mean_iou, mean_acc = evaluate_model(model, test_loader, DEVICE, NUM_CLASSES, USE_VAE)
    print(f"\nTest Results:")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    
    # Visualize
    visualize_predictions(model, test_dataset, DEVICE, num_samples=4, is_vae=USE_VAE)