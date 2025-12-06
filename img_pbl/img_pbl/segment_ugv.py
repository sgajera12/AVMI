"""
INFERENCE SCRIPT FOR YOUR UGV DATA

Use this to segment your UGV images after training
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train_cpu_ugv import LightweightSegmentation

# Rellis-3D color map for visualization
RELLIS_COLORS = {
    0: [0, 0, 0],          # void
    1: [108, 64, 20],      # dirt
    3: [0, 102, 0],        # grass
    4: [0, 255, 0],        # tree
    5: [0, 153, 153],      # pole
    6: [0, 128, 255],      # water
    7: [0, 0, 255],        # sky
    8: [255, 255, 0],      # vehicle
    9: [255, 0, 127],      # object
    10: [64, 64, 64],      # asphalt
    12: [255, 0, 0],       # building
    15: [102, 0, 0],       # log
    17: [204, 153, 255],   # person
    18: [102, 0, 204],     # fence
    19: [255, 153, 204],   # bush
    23: [170, 170, 170],   # concrete
    27: [41, 121, 255],    # barrier
    31: [134, 255, 239],   # puddle
    33: [99, 66, 34],      # mud
    34: [110, 22, 138],    # rubble
}

def load_model(checkpoint_path, device='cpu'):
    """Load trained model"""
    model = LightweightSegmentation(num_classes=35).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def segment_image(model, image_path, img_size=(128, 128), device='cpu'):
    """Segment a single UGV image"""
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Resize prediction back to original size
    pred_resized = Image.fromarray(pred.astype(np.uint8))
    pred_resized = pred_resized.resize(original_size, Image.NEAREST)
    pred_resized = np.array(pred_resized)
    
    return pred_resized, image


def visualize_segmentation(image, prediction, save_path=None):
    """Visualize segmentation result"""
    # Create colored segmentation mask
    h, w = prediction.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in RELLIS_COLORS.items():
        mask = prediction == class_id
        colored_mask[mask] = color
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original UGV Image')
    axes[0].axis('off')
    
    axes[1].imshow(colored_mask)
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.array(image) * 0.5 + colored_mask * 0.5
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def segment_video_frames(model, video_frames_dir, output_dir, device='cpu'):
    """Segment all frames from your UGV video"""
    video_frames_dir = Path(video_frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all image frames
    image_files = sorted(list(video_frames_dir.glob('*.jpg')) + 
                        list(video_frames_dir.glob('*.png')))
    
    print(f"Found {len(image_files)} frames to segment")
    
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
        
        pred, img = segment_image(model, img_path, device=device)
        
        # Save visualization
        save_path = output_dir / f"segmented_{img_path.stem}.png"
        visualize_segmentation(img, pred, save_path=save_path)
        
        # Also save just the prediction mask
        pred_path = output_dir / f"mask_{img_path.stem}.png"
        Image.fromarray(pred.astype(np.uint8)).save(pred_path)
    
    print(f"\nDone! Results saved to {output_dir}")


def analyze_classes(model, image_path, device='cpu'):
    """Analyze which classes are detected in your UGV image"""
    pred, _ = segment_image(model, image_path, device=device)
    
    unique_classes = np.unique(pred)
    
    class_names = {
        0: 'void', 1: 'dirt', 3: 'grass', 4: 'tree', 5: 'pole',
        6: 'water', 7: 'sky', 8: 'vehicle', 9: 'object', 10: 'asphalt',
        12: 'building', 15: 'log', 17: 'person', 18: 'fence', 19: 'bush',
        23: 'concrete', 27: 'barrier', 31: 'puddle', 33: 'mud', 34: 'rubble'
    }
    
    print("\nDetected classes in your UGV image:")
    print("-" * 50)
    for class_id in unique_classes:
        pixel_count = np.sum(pred == class_id)
        percentage = (pixel_count / pred.size) * 100
        class_name = class_names.get(class_id, f'unknown_{class_id}')
        print(f"{class_name:15s} (ID {class_id:2d}): {percentage:5.2f}% ({pixel_count} pixels)")
    print("-" * 50)


if __name__ == '__main__':
    # Configuration
    MODEL_PATH = 'ugv_segmentation_epoch5.pth'  # Path to your trained model
    UGV_IMAGE_PATH = '/path/to/your/ugv/image.jpg'  # Single image to test
    UGV_FRAMES_DIR = '/path/to/your/ugv/video/frames'  # Directory with video frames
    OUTPUT_DIR = './segmentation_results'
    
    DEVICE = 'cpu'  # Use CPU since CUDA isn't working
    
    # Load model
    model = load_model(MODEL_PATH, device=DEVICE)
    
    print("\n" + "="*60)
    print("OPTION 1: Segment a single test image")
    print("="*60)
    pred, img = segment_image(model, UGV_IMAGE_PATH, device=DEVICE)
    visualize_segmentation(img, pred, save_path='test_segmentation.png')
    analyze_classes(model, UGV_IMAGE_PATH, device=DEVICE)
    
    print("\n" + "="*60)
    print("OPTION 2: Segment all video frames")
    print("="*60)
    # Uncomment to process all frames:
    # segment_video_frames(model, UGV_FRAMES_DIR, OUTPUT_DIR, device=DEVICE)
