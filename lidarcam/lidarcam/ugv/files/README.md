# UGV LiDAR-Camera Fusion for Unreal Engine Dataset

This package performs sensor fusion between LiDAR and camera data from your UGV in Unreal Engine. It projects 3D LiDAR points onto 2D camera images, creating two visualization outputs.

## 📋 Table of Contents
- [What This Does](#what-this-does)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Understanding the Output](#understanding-the-output)
- [Calibration Explained](#calibration-explained)
- [Troubleshooting](#troubleshooting)

---

## 🎯 What This Does

### Input:
- **Camera images**: RGB photos from your front-facing camera (640×480, 90° FOV)
- **LiDAR scans**: 3D point clouds from your LiDAR sensor

### Output:
1. **Depth-colored projection**: Camera image with LiDAR points overlaid, colored by distance
   - Blue = Close objects
   - Green = Medium distance
   - Red = Far objects

2. **RGB-colored projection**: Black background with LiDAR points colored by the image
   - Shows which parts of the image have LiDAR data
   - Helps verify calibration accuracy

---

## 🔬 How It Works

### The Big Picture

Imagine you're standing with two cameras: one sees in 3D (LiDAR) and one sees in 2D with colors (camera). Your goal is to combine them so you know both the distance AND color of everything you see.

```
        LiDAR Point in 3D                Camera Image (2D)
              
         Z (up)                           
         ↑                                  Y (down)
         |  • Point                         ↓
         | /                             -------
         |/___→ X (forward)              |     |
        /                                |  •  | ← Projected point
       Y (left)                          |     |
                                         -------
                                         → X (right)
```

### Step-by-Step Process

#### Step 1: Load Calibration
**What**: Load camera intrinsics (how camera sees) and extrinsics (where sensors are)

**Why**: Every camera is different, and sensors are mounted in different positions. We need to know these specifics.

**How**: 
- **Intrinsics**: Calculate from 90° FOV and 640×480 resolution
  - Focal length: 320 pixels (determines zoom/magnification)
  - Center point: (320, 240) - middle of the image
  
- **Extrinsics**: Calculate from Unreal Engine positions
  - Your camera is ~1.88m FORWARD of LiDAR
  - Your camera is ~0.76m BELOW LiDAR

#### Step 2: Read Bag File
**What**: Load synchronized camera and LiDAR messages from your recording

**Why**: We need matching pairs - a camera image and LiDAR scan from the same moment

**How**: 
1. Open the SQLite database (.db3 file)
2. Find camera and LiDAR topics
3. Match messages by timestamp (within 50ms)

#### Step 3: Transform Coordinates
**What**: Convert LiDAR points from LiDAR frame to Camera frame

**Why**: LiDAR measures positions relative to itself, camera sees things relative to itself. We need everything in camera's view.

**Formula**:
```
P_camera = R × P_lidar + t

Where:
- P_lidar = [x, y, z] in LiDAR coordinates
- R = Rotation matrix (identity in your case)
- t = Translation vector [1.878, 0.0, -0.764] meters
- P_camera = [x', y', z'] in Camera coordinates
```

**Example**:
```python
LiDAR point: [5.0, 0.5, 2.0] meters
             (5m forward, 0.5m left, 2m up from LiDAR)

After transformation:
Camera point: [6.878, 0.5, 1.236] meters
              (6.878m forward, 0.5m left, 1.236m up from Camera)
```

#### Step 4: Filter Points
**What**: Keep only points in front of the camera

**Why**: Cameras can only see what's in front of them (Z > 0)

**How**: Remove any points with negative or very small Z values

#### Step 5: Project to 2D
**What**: Convert 3D camera coordinates to 2D image pixel coordinates

**Why**: The 3D world needs to be "flattened" onto the 2D image plane

**Formula**:
```
[u]       [X/Z]
[v] = K × [Y/Z]
[1]       [ 1 ]

Where:
K = Camera matrix [fx  0  cx]
                  [ 0 fy  cy]
                  [ 0  0   1]

u, v = pixel coordinates in image
X, Y, Z = 3D position in camera frame
```

**Visual Example**:
```
3D Point in camera: [1.0, 0.5, 5.0] meters
                    (1m right, 0.5m up, 5m forward)

Step 1: Normalize by depth (divide by Z):
[1.0/5.0, 0.5/5.0] = [0.2, 0.1]

Step 2: Apply camera matrix:
u = fx × 0.2 + cx = 320 × 0.2 + 320 = 384
v = fy × 0.1 + cy = 320 × 0.1 + 240 = 272

Result: Pixel (384, 272) in image
```

#### Step 6: Color and Draw
**What**: Assign colors to projected points and draw them

**For depth-colored output**:
- Calculate depth (Z value) for each point
- Normalize: `(Z - Z_min) / (Z_max - Z_min)` → value between 0 and 1
- Apply JET colormap: 0→Blue, 0.5→Green, 1→Red

**For RGB-colored output**:
- Look up the pixel color at (u, v) in original image
- Draw point with that color on black background

---

## 📦 Installation

### Prerequisites
```bash
# ROS2 (Humble or newer)
# Python 3.8+
# OpenCV
# NumPy
```

### Step 1: Create ROS2 Workspace
```bash
mkdir -p ~/ugv_ws/src
cd ~/ugv_ws/src
```

### Step 2: Create Package
```bash
ros2 pkg create ugv_fusion --build-type ament_python --dependencies rclpy sensor_msgs std_msgs cv_bridge
```

### Step 3: Copy Files
```bash
# Copy the node script
cp ugv_fusion_node.py ~/ugv_ws/src/ugv_fusion/ugv_fusion/
chmod +x ~/ugv_ws/src/ugv_fusion/ugv_fusion/ugv_fusion_node.py

# Copy the launch file
mkdir -p ~/ugv_ws/src/ugv_fusion/launch
cp ugv_fusion_launch.py ~/ugv_ws/src/ugv_fusion/launch/
```

### Step 4: Update setup.py
Edit `~/ugv_ws/src/ugv_fusion/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'ugv_fusion'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='UGV LiDAR-Camera Fusion',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ugv_fusion_node = ugv_fusion.ugv_fusion_node:main',
        ],
    },
)
```

### Step 5: Build
```bash
cd ~/ugv_ws
colcon build --packages-select ugv_fusion
source install/setup.bash
```

---

## 🚀 Usage

### Quick Start
```bash
# Terminal 1: Start the fusion node
ros2 run ugv_fusion ugv_fusion_node --ros-args -p bag_file:=/path/to/your/run1_lidar_camera_0.db3

# Terminal 2: Launch RViz to visualize
rviz2
```

### Using Launch File
```bash
ros2 launch ugv_fusion ugv_fusion_launch.py
```

### Custom Parameters
```bash
# Use different bag file
ros2 launch ugv_fusion ugv_fusion_launch.py bag_file:=/path/to/other_bag.db3

# Change playback speed (5 Hz instead of 10 Hz)
ros2 launch ugv_fusion ugv_fusion_launch.py republish_rate_hz:=5.0
```

### View Topics
```bash
# List available topics
ros2 topic list

# Expected topics:
# /camera/image_raw                  - Original camera image
# /lidar/points2                     - Original LiDAR point cloud
# /fusion/projected_image            - Depth-colored fusion
# /fusion/color_projection_image     - RGB-colored projection
```

### Visualize in RViz
```bash
rviz2
```

Add displays:
1. **Image** → Topic: `/camera/image_raw`
2. **Image** → Topic: `/fusion/projected_image`
3. **Image** → Topic: `/fusion/color_projection_image`
4. **PointCloud2** → Topic: `/lidar/points2`

---

## 🎨 Understanding the Output

### Output 1: Depth-Colored Projection
**File**: Published on `/fusion/projected_image`

**What you see**: Original camera image with colored dots representing LiDAR points

**Color meaning**:
- 🔵 **Blue**: Close objects (e.g., 0-5 meters)
- 🟢 **Green**: Medium distance (e.g., 5-15 meters)
- 🔴 **Red**: Far objects (e.g., 15-30+ meters)

**Use cases**:
- Quick depth perception
- Obstacle detection
- Distance estimation to objects

**Example interpretation**:
```
If you see:
- Blue dots on a tree trunk → Tree is close (~2-3m)
- Red dots on distant road → Road extends far (~20m+)
- Green dots on a vehicle → Vehicle is at medium range (~8-10m)
```

### Output 2: RGB-Colored Projection
**File**: Published on `/fusion/color_projection_image`

**What you see**: Black background with dots colored by the original image

**Color meaning**: Each dot's color matches the pixel color from the camera image

**Use cases**:
- Verify calibration accuracy
  - If colors match objects → Good calibration ✅
  - If colors are offset → Bad calibration ❌
- See LiDAR coverage patterns
- Identify which objects have depth data

**Example interpretation**:
```
If you see:
- Green dots in grass area → LiDAR detecting grass ✅
- Brown dots on tree trunk → LiDAR detecting tree ✅
- Green dots on tree trunk → Misalignment! ❌ (Calibration issue)
```

---

## 🔧 Calibration Explained

### Camera Intrinsics (How the Camera Sees)

**Camera Matrix (K)**:
```
K = [fx  0  cx]
    [ 0 fy  cy]
    [ 0  0   1]
```

**Your values**:
- `fx = fy = 320` pixels (focal length)
- `cx = 320` pixels (horizontal center)
- `cy = 240` pixels (vertical center)

**How we got these**:
1. FOV = 90 degrees
2. Formula: `fx = (width/2) / tan(FOV/2)`
3. Calculation: `fx = (640/2) / tan(45°) = 320 / 1 = 320`

**What this means**:
- 90° FOV is a "normal" field of view (like human peripheral vision)
- Focal length of 320 means a point 1 meter away projects to ~320 pixels
- Center at (320, 240) means image is symmetric

### Extrinsics (Where Sensors Are Located)

**Translation vector (t)**:
```
t = [1.878, 0.0, -0.764] meters
```

**What this means**:
- Camera is 1.878m **FORWARD** of LiDAR (X-axis)
- Camera is 0.0m **LEFT/RIGHT** of LiDAR (Y-axis)  
- Camera is 0.764m **BELOW** LiDAR (Z-axis)

**Physical interpretation**:
```
Side view of your UGV:

       LiDAR •  ← LiDAR sensor (higher up)
              |
              | 0.764m down
              |
              ↓
     Camera • → ← Camera sensor (lower, looking forward)
              
       ← 1.878m forward →
```

**Rotation matrix (R)**:
```
R = [1 0 0]
    [0 1 0]  (Identity matrix)
    [0 0 1]
```

**What this means**:
- Both sensors point in the same direction (no rotation)
- Both use the same orientation (aligned axes)

---

## 🔍 Troubleshooting

### Problem: "No synchronized frames found"

**Possible causes**:
1. Bag file doesn't contain both camera and LiDAR data
2. Topic names are wrong
3. Timestamps are too far apart

**Solutions**:
```bash
# Check what topics exist in your bag
ros2 bag info /path/to/your/bag.db3

# Check topic names in code match bag
# Edit ugv_fusion_node.py lines 111-112:
camera_topic_name = '/camera/image_raw'  # Match your actual topic
lidar_topic_name = '/lidar/points2'      # Match your actual topic

# If timestamps are very different, increase tolerance
# Edit line 271:
sync_tolerance = 100_000_000  # Try 100ms instead of 50ms
```

### Problem: "Points project to wrong locations"

**Possible causes**:
1. Incorrect extrinsic calibration
2. Wrong coordinate system assumptions
3. Unreal units vs. ROS units mismatch

**Solutions**:
```python
# 1. Verify positions from Unreal Engine
# Double-check the values in your Unreal setup

# 2. Try adjusting translation
# In _setup_calibration(), try:
self.t_l2c = np.array([1.878, 0.0, -0.764], dtype=np.float32)

# If still wrong, try different coordinate conversions:
# Option A: Flip X
self.t_l2c = np.array([-1.878, 0.0, -0.764], dtype=np.float32)

# Option B: Flip Z
self.t_l2c = np.array([1.878, 0.0, 0.764], dtype=np.float32)

# Option C: Both
self.t_l2c = np.array([-1.878, 0.0, 0.764], dtype=np.float32)
```

### Problem: "No points visible on image"

**Possible causes**:
1. All points behind camera
2. FOV too narrow
3. Points outside image bounds

**Solutions**:
```python
# 1. Check if points are in front
# Add debug print in _project_points_depth_colored():
print(f"Total points: {len(lidar_xyz)}")
print(f"Points in front: {front.sum()}")
print(f"Points in image: {keep.sum()}")

# 2. Check point cloud range
print(f"X range: [{lidar_xyz[:, 0].min():.2f}, {lidar_xyz[:, 0].max():.2f}]")
print(f"Y range: [{lidar_xyz[:, 1].min():.2f}, {lidar_xyz[:, 1].max():.2f}]")
print(f"Z range: [{lidar_xyz[:, 2].min():.2f}, {lidar_xyz[:, 2].max():.2f}]")

# 3. Adjust filtering threshold
# In _project_points_depth_colored(), line 418:
front = Z > 0.01  # Try smaller threshold (was 0.1)
```

### Problem: "Colors don't match objects"

**This means calibration is off!**

**Solutions**:
1. Re-check sensor positions in Unreal Engine
2. Verify coordinate system conversions
3. Try the coordinate flip options above
4. Consider that Unreal units might be in different scale

### Problem: "Code runs but images are blank"

**Possible causes**:
1. Image encoding issue
2. Points filtered out
3. Visualization not showing

**Solutions**:
```bash
# Check if messages are being published
ros2 topic echo /fusion/projected_image --once

# Check message rate
ros2 topic hz /fusion/projected_image

# Try viewing with rqt_image_view
rqt_image_view /fusion/projected_image
```

---

## 📊 Performance Tips

### Adjust Playback Speed
```bash
# Slower (easier to see, less CPU)
ros2 launch ugv_fusion ugv_fusion_launch.py republish_rate_hz:=1.0

# Faster (real-time, more CPU)
ros2 launch ugv_fusion ugv_fusion_launch.py republish_rate_hz:=30.0
```

### Reduce Point Cloud Size
If too slow, downsample LiDAR in the code:
```python
# In _extract_points_from_cloud(), add:
if len(points) > 10000:
    # Keep every 2nd point
    points = points[::2]
```

---

## 🎓 Learning Resources

### Understanding Camera Calibration:
- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- Explains intrinsics, extrinsics, distortion

### Understanding Coordinate Transformations:
- [ROS TF2 Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html)
- How different coordinate frames relate

### Understanding Sensor Fusion:
- Search: "LiDAR camera fusion tutorial"
- Explains why and how to combine sensors

---

## 📝 Notes

### Coordinate Systems

**Unreal Engine (Left-handed)**:
- X: Forward
- Y: Right
- Z: Up

**ROS (Right-handed)**:
- X: Forward
- Y: Left
- Z: Up

**Conversion**: Flip Y axis (Unreal_Y → -ROS_Y)

### Units
- Unreal Engine positions: **Centimeters**
- ROS positions: **Meters**
- Conversion: Divide by 100

### Assumptions
1. Both sensors have 0° rotation in Unreal (looking forward)
2. Positions are relative to robot base frame
3. LiDAR returns points in its own frame
4. Camera has no distortion (or it's already corrected)

---

## 🆘 Still Having Issues?

### Debug Checklist:
- [ ] Bag file exists and is readable
- [ ] Topic names match between code and bag
- [ ] Camera and LiDAR timestamps are synchronized
- [ ] Sensor positions are correct in Unreal
- [ ] Coordinate conversions are correct
- [ ] FOV and image size are correct
- [ ] Points are projecting in front of camera

### Get Help:
1. Check ROS logs: `ros2 run ugv_fusion ugv_fusion_node`
2. Visualize point cloud separately in RViz
3. Record a short test sequence (5 seconds) to debug faster

---

**Good luck with your sensor fusion! 🚀**
