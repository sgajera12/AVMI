#!/usr/bin/env python3
"""
Rellis-3D Data Loader - FIXED VERSION

Fixes:
1. Frame matching by actual frame numbers
2. Correct color mapping for Rellis-3D
3. Better LiDAR feature visualization
4. No mirrored/flipped data
5. Proper height/density/intensity normalization
"""

import numpy as np
import cv2
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset, DataLoader
import re


class Rellis3DDataset(Dataset):
    """Rellis-3D Dataset with RGB + LiDAR fusion"""
    
    def __init__(self, 
                 base_path='/home/pinaka/dataset/rellis3d/Rellis-3D',
                 sequence='00000',
                 split='train',
                 img_size=(600, 960),
                 augment=True):
        
        self.base = Path(base_path)
        self.seq = sequence
        self.split = split
        self.img_size = img_size
        
        # Rellis-3D official color map
        # Rellis-3D official color map
        self.classes = {
            0: 'void', 1: 'dirt', 3: 'grass', 4: 'tree', 5: 'pole',
            6: 'water', 7: 'sky', 8: 'vehicle', 9: 'object', 10: 'asphalt',
            12: 'building', 15: 'log', 17: 'person', 18: 'fence', 19: 'bush',
            23: 'concrete', 27: 'barrier', 31: 'puddle', 33: 'mud', 34: 'rubble'
        }
        
        # Official Rellis-3D color map (RGB format for consistency)
        # These match the ontology.png hex codes exactly
        self.color_map = {
            0: [0, 0, 0],          # void - #000000
            1: [108, 64, 20],      # dirt - #6c4014
            3: [0, 102, 0],        # grass - #006600
            4: [0, 255, 0],        # tree - #00ff00
            5: [0, 153, 153],      # pole - #009999
            6: [0, 128, 255],      # water - #0080ff
            7: [0, 0, 255],        # sky - #0000ff
            8: [255, 255, 0],      # vehicle - #ffff00
            9: [255, 0, 127],      # object - #ff007f
            10: [64, 64, 64],       # asphalt - #404040
            12: [255, 0, 0],       # building - #ff0000
            15: [102, 0, 0],       # log - #660000
            17: [204, 153, 255],   # person - #cc99ff
            18: [102, 0, 204],     # fence - #6600cc
            19: [255, 153, 204],   # bush - #ff99cc
            23: [170, 170, 170],   # concrete - #aaaaaa
            27: [41, 121, 255],    # barrier - #2979ff
            31: [134, 255, 239],   # puddle - #86ffef
            33: [99, 66, 34],      # mud - #634222
            34: [110, 22, 138],    # rubble - #6e168a
        }
        
        # Also store hex codes for verification
        self.hex_codes = {
            0: '#000000', 1: '#6c4014', 3: '#006600', 4: '#00ff00', 5: '#009999',
            6: '#0080ff', 7: '#0000ff', 8: '#ffff00', 9: '#ff007f', 10: '#404040',
            12: '#ff0000', 15: '#660000', 17: '#cc99ff', 18: '#6600cc', 19: '#ff99cc',
            23: '#aaaaaa', 27: '#2979ff', 31: '#86ffef', 33: '#634222', 34: '#6e168a'
        }
        
 
        
        self.num_classes = len(self.classes)
        
        # Paths
        self.img_dir = self.base / f"image/{self.seq}/pylon_camera_node"
        self.lidar_dir = self.base / f"lidar/{self.seq}/os1_cloud_node_kitti_bin"
        self.label_dir = self.base / f"image_id/{self.seq}/pylon_camera_node_label_id"
        
        # Match files by frame number
        self.samples = self._match_files()
        
        # Train/val split
        n_total = len(self.samples)
        n_train = int(n_total * 0.8)
        
        if split == 'train':
            self.samples = self.samples[:n_train]
        else:
            self.samples = self.samples[n_train:]
        
        # Load calibration
        self._load_calibration()
        
        self.augment = augment and split == 'train'
        
        print(f"Rellis-3D {split} dataset:")
        print(f"  Frames: {len(self.samples)}")
        print(f"  Size: {img_size}")
        print(f"  Augment: {self.augment}")
    
    def _match_files(self):
        """Match files by frame number in filename"""
        def get_frame_num(filepath):
            match = re.search(r'frame(\d+)', filepath.name) or re.search(r'(\d+)', filepath.name)
            return int(match.group(1)) if match else 0
        
        img_files = sorted(self.img_dir.glob("*.jpg"))
        lidar_files = sorted(self.lidar_dir.glob("*.bin"))
        label_files = sorted(self.label_dir.glob("*.png"))
        
        img_dict = {get_frame_num(f): f for f in img_files}
        lidar_dict = {get_frame_num(f): f for f in lidar_files}
        label_dict = {get_frame_num(f): f for f in label_files}
        
        common_frames = sorted(set(img_dict.keys()) & set(lidar_dict.keys()) & set(label_dict.keys()))
        
        samples = [{
            'frame_num': fn,
            'image': img_dict[fn],
            'lidar': lidar_dict[fn],
            'label': label_dict[fn]
        } for fn in common_frames]
        
        print(f"Matched {len(samples)} frames")
        return samples
    
    def _load_calibration(self):
        """Load calibration"""
        calib_file = self.base / f"calibration/{self.seq}/transforms.yaml"
        with open(calib_file, 'r') as f:
            tf = yaml.safe_load(f)["os1_cloud_node-pylon_camera_node"]
        
        q, t = tf["q"], tf["t"]
        R_c2l = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32)
        t_c2l = np.array([t["x"], t["y"], t["z"]], dtype=np.float32)
        
        self.R_l2c = R_c2l.T
        self.t_l2c = -R_c2l.T @ t_c2l
        
        intr_file = self.base / f"intrinsic/{self.seq}/camera_info.txt"
        fx, fy, cx, cy = np.loadtxt(intr_file)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load data
        img = cv2.imread(str(sample['image']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pts = np.fromfile(sample['lidar'], dtype=np.float32).reshape(-1, 4)
        xyz = pts[:, :3]
        intensity = pts[:, 3]
        
        label = cv2.imread(str(sample['label']), cv2.IMREAD_UNCHANGED)
        label = np.clip(label, 0, self.num_classes - 1)
        
        # Create LiDAR features at original resolution
        lidar_features = self._create_lidar_features(xyz, intensity, img.shape[:2])
        
        # Resize everything
        img_resized = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        label_resized = cv2.resize(label, (self.img_size[1], self.img_size[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        lidar_features_resized = np.zeros((3, self.img_size[0], self.img_size[1]), dtype=np.float32)
        for i in range(3):
            lidar_features_resized[i] = cv2.resize(lidar_features[i], 
                                                    (self.img_size[1], self.img_size[0]))
        
        # Simple augmentation (NO FLIP - causes mirror issue!)
        if self.augment:
            if np.random.rand() > 0.7:
                factor = np.random.uniform(0.8, 1.2)
                img_resized = np.clip(img_resized * factor, 0, 255).astype(np.uint8)
        
        # To tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        lidar_tensor = torch.from_numpy(lidar_features_resized).float()
        rgb_lidar = torch.cat([img_tensor, lidar_tensor], dim=0)
        
        label_tensor = torch.from_numpy(label_resized).long()
        
        return rgb_lidar, label_tensor, sample['frame_num']
    
    def _create_lidar_features(self, xyz, intensity, img_shape):
        """Create height/density/intensity maps"""
        h, w = img_shape
        
        height_map = np.zeros((h, w), dtype=np.float32)
        density_map = np.zeros((h, w), dtype=np.float32)
        intensity_map = np.zeros((h, w), dtype=np.float32)
        intensity_count = np.zeros((h, w), dtype=np.float32)
        
        # Transform and project
        Pc = (self.R_l2c @ xyz.T + self.t_l2c.reshape(3, 1)).T
        Z = Pc[:, 2]
        front = Z > 0
        
        uv = (self.K @ (Pc.T / np.maximum(Z, 1e-6))).T
        u, v = uv[:, 0], uv[:, 1]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        valid = front & inside
        u_v = u[valid].astype(np.int32)
        v_v = v[valid].astype(np.int32)
        z_v = Pc[valid, 2]
        i_v = intensity[valid]
        
        # Fill maps
        for uu, vv, zz, ii in zip(u_v, v_v, z_v, i_v):
            height_map[vv, uu] = max(height_map[vv, uu], zz)
            density_map[vv, uu] += 1
            intensity_map[vv, uu] += ii
            intensity_count[vv, uu] += 1
        
        # Normalize
        mask = intensity_count > 0
        intensity_map[mask] /= intensity_count[mask]
        
        # Debug info
        print(f"  LiDAR stats: height={height_map.min():.2f} to {height_map.max():.2f}, "
              f"density max={density_map.max():.0f}, valid pixels={valid.sum()}")
        
        # Better normalization
        height_map = np.clip(height_map, -5, 50)
        height_map = (height_map + 5) / 55.0  # -5→50 becomes 0→1
        density_map = np.log1p(density_map) / np.log1p(20)
        intensity_map = intensity_map / (intensity_map.max() + 1e-6)
        
        return np.stack([height_map, density_map, intensity_map], axis=0)


def visualize_sample(rgb_lidar, label, frame_num, dataset):
    """Visualize with correct colors"""
    import matplotlib.pyplot as plt
    
    rgb = rgb_lidar[:3].permute(1, 2, 0).numpy()
    height = rgb_lidar[3].numpy()
    density = rgb_lidar[4].numpy()
    intensity = rgb_lidar[5].numpy()
    label = label.numpy()
    
    # Denormalize RGB
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = np.clip(rgb * std + mean, 0, 1)
    
    # Create label visualization with correct colors (RGB, not BGR!)
    label_color = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id, color in dataset.color_map.items():
        mask = label == class_id
        label_color[mask] = color  # Already in RGB
    
    # Better visualizations with actual data range
    # Height: already normalized to [0, 1]
    height_vis = height
    
    # Density: apply colormap to see variation
    density_nonzero = density > 0
    if density_nonzero.sum() > 0:
        density_vis = np.zeros_like(density)
        density_vis[density_nonzero] = density[density_nonzero]
        density_vis = plt.cm.hot(density_vis)[:, :, :3]
    else:
        density_vis = np.zeros((*density.shape, 3))
    
    # Intensity: apply grayscale colormap
    intensity_nonzero = intensity > 0
    if intensity_nonzero.sum() > 0:
        intensity_vis = np.zeros_like(intensity)
        intensity_vis[intensity_nonzero] = intensity[intensity_nonzero]
        intensity_vis = plt.cm.gray(intensity_vis)[:, :, :3]
    else:
        intensity_vis = np.zeros((*intensity.shape, 3))
    
    # RGB with LiDAR overlay
    rgb_with_lidar = (rgb * 255).astype(np.uint8).copy()
    
    # Plot
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    ax1.set_title(f'RGB Image (Frame {frame_num})', fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_with_lidar)
    y_coords, x_coords = np.where(density > 0.01)
    if len(x_coords) > 0:
        ax2.scatter(x_coords[::5], y_coords[::5], c=density[density > 0.01][::5], 
                   cmap='hot', s=3, alpha=0.6)
    ax2.set_title('RGB + LiDAR Overlay', fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(height_vis, cmap='viridis', vmin=0, vmax=1)
    ax3.set_title('Height Map', fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cbar3.set_label('Normalized Height', rotation=270, labelpad=15)
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(density_vis)
    ax4.set_title('Density Map', fontweight='bold')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(intensity_vis)
    ax5.set_title('Intensity Map', fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(label_color)
    ax6.set_title('Ground Truth', fontweight='bold')
    ax6.axis('off')
    
    # Class distribution
    ax7 = fig.add_subplot(gs[2, :2])
    unique, counts = np.unique(label, return_counts=True)
    class_names = [dataset.classes[c] for c in unique]
    colors_bar = [np.array(dataset.color_map.get(c, [128, 128, 128])) / 255.0 for c in unique]  # RGB already
    
    bars = ax7.bar(range(len(unique)), counts, color=colors_bar, edgecolor='black')
    ax7.set_xticks(range(len(unique)))
    ax7.set_xticklabels(class_names, rotation=45, ha='right')
    ax7.set_title('Class Distribution', fontweight='bold')
    ax7.set_ylabel('Pixel Count')
    ax7.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., h, f'{count}', 
                ha='center', va='bottom', fontsize=9)
    
    # Legend
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    ax8.set_title('Class Legend', fontweight='bold')
    
    y_pos = 0.95
    for class_id in unique[:12]:
        if class_id in dataset.color_map:
            color = np.array(dataset.color_map[class_id]) / 255.0  # RGB, already correct
            ax8.add_patch(plt.Rectangle((0.1, y_pos), 0.15, 0.05, 
                                       facecolor=color, edgecolor='black'))
            ax8.text(0.3, y_pos + 0.025, dataset.classes[class_id], 
                    va='center', fontsize=10)
            y_pos -= 0.07
    
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'sample_frame_{frame_num}.png', dpi=200, bbox_inches='tight')
    print(f"✅ Saved: sample_frame_{frame_num}.png")
    plt.show()


def test_dataloader():
    """Test the fixed dataloader"""
    print("="*70)
    print("TESTING FIXED RELLIS-3D DATALOADER")
    print("="*70)
    
    dataset = Rellis3DDataset(
        base_path='/home/pinaka/dataset/rellis3d/Rellis-3D',
        split='train',
        augment=False  # No augment for testing
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load sample
    rgb_lidar, label, frame_num = dataset[0]
    print(f"\nFrame {frame_num}:")
    print(f"  RGB+LiDAR: {rgb_lidar.shape}")
    print(f"  Label: {label.shape}")
    print(f"  Classes: {torch.unique(label).numpy()}")
    
    # Visualize
    visualize_sample(rgb_lidar, label, frame_num, dataset)
    
    print("\n✅ Test complete!")


if __name__ == '__main__':
    test_dataloader()