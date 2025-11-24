# 🎯 Coordinate Systems Reference - Visual Guide

This document helps you understand and debug coordinate transformations.

---

## 🌍 Coordinate System Comparison

### Unreal Engine (Left-Handed)
```
        Z (Up)
        ↑
        |
        |___→ X (Forward)
       /
      Y (Right)
```

**Example in Unreal:**
- Forward = Positive X
- Right = Positive Y
- Up = Positive Z

**Your sensor positions (from Unreal):**
```
LiDAR:  X=-31.08,  Y=-0.00,  Z=179.71 cm
Camera: X=156.73,  Y=0.00,   Z=103.34 cm
```

### ROS (Right-Handed)
```
        Z (Up)
        ↑
        |
        |___→ X (Forward)
       /
      Y (Left)  ← Notice: LEFT, not right!
```

**Example in ROS:**
- Forward = Positive X
- Left = Positive Y (opposite of Unreal!)
- Up = Positive Z

**After conversion:**
```
Translation: [1.878, 0.0, -0.764] meters
```

---

## 🔄 Conversion Rule

**Simple rule for Unreal → ROS:**
```
ROS_X =  Unreal_X   (forward stays forward)
ROS_Y = -Unreal_Y   (right becomes left → flip!)
ROS_Z =  Unreal_Z   (up stays up)
```

**Example:**
```
Unreal point: [5.0, 2.0, 1.0]
             (5m forward, 2m right, 1m up)

ROS point:    [5.0, -2.0, 1.0]
             (5m forward, 2m LEFT, 1m up)
```

---

## 🚗 Your UGV Physical Layout

### Top View (Looking Down):
```
        Forward →
        
    [LiDAR]  ← Back here
        |
        |
        | 1.878m
        |
        ↓
    [Camera] ← Front here
```

### Side View (Looking from Left):
```
    • LiDAR    ← Higher (at Z=1.797m)
    |
    | 0.764m down
    |
    ↓
    • Camera   ← Lower (at Z=1.033m)
    
    Both looking forward →
```

---

## 📐 Transformation Math (Visual)

### LiDAR Point → Camera Frame

**Step 1: Apply Translation**
```
LiDAR point:     [3.0, 0.5, 0.0]
                 (3m forward, 0.5m left, 0m up from LiDAR)

Add translation: +[1.878, 0.0, -0.764]
                 (camera offset from LiDAR)

Camera point:    [4.878, 0.5, -0.764]
                 (4.878m forward, 0.5m left, 0.764m down from camera)

Check: Z = -0.764 is NEGATIVE!
→ This point is BELOW the camera
→ Camera looking forward won't see it
→ Point will be filtered out
```

**Step 2: Project to Image (for points with Z > 0)**
```
Camera point:    [2.0, 0.5, 5.0]
                 (2m right, 0.5m up, 5m forward)

Divide by depth: [2.0/5.0, 0.5/5.0] = [0.4, 0.1]

Apply K matrix:
u = fx × 0.4 + cx = 320 × 0.4 + 320 = 448 pixels
v = fy × 0.1 + cy = 320 × 0.1 + 240 = 272 pixels

Result: Pixel (448, 272)
```

---

## 🎨 Visual Examples

### Example 1: Point in Front
```
LiDAR measures: [5, 0, 0] "Something 5m ahead"
                
After transform: [6.878, 0, -0.764]
                 Z < 0 → BEHIND camera vertically
                 
Result: NOT visible (filtered out)

Why? Camera is looking forward and DOWN relative to LiDAR
If LiDAR sees something at Z=0 (its height),
that's BELOW the camera's height.
```

### Example 2: Point Above
```
LiDAR measures: [5, 0, 2] "Something 5m ahead, 2m up"
                
After transform: [6.878, 0, 1.236]
                 Z > 0 → In front!
                 
Project: u = K × [6.878/1.236, 0] ≈ [1780, 240]
         u=1780 is > 640 → Outside image (right edge)
         
Result: NOT visible (outside FOV)
```

### Example 3: Point in View
```
LiDAR measures: [5, 0, 1] "Something 5m ahead, 1m up"
                
After transform: [6.878, 0, 0.236]
                 Z > 0 → In front!
                 
Project: u = K × [6.878/0.236, 0] = K × [29.1, 0]
         u = 320 × 29.1 + 320 = 9632 (way off screen)
         
Result: NOT visible

Hmm, still outside! Let's try closer to camera level...
```

### Example 4: Realistic Point
```
LiDAR measures: [5, 0.2, 0.8] "Object 5m ahead, slightly left and up"
                
After transform: [6.878, 0.2, 0.036]
                 Z = 0.036 > 0 → In front! (barely)
                 
Project: u = K × [6.878/0.036, 0.2/0.036]
         = K × [191.1, 5.6]
         u = 320 × 191.1 + 320 = 61472 (off screen)
         
Wait, still off. Need points closer to camera Z=0...
```

**Lesson:** Most LiDAR points near Z=0 (LiDAR height) will be:
- Below camera (Z - 0.764 < 0) → Behind camera
- Or barely in front but at steep angle → Outside FOV

**You'll see points from:**
- Objects above LiDAR height (trees, buildings)
- Ground in front (which camera sees)
- Objects at similar height to camera

---

## 🔧 Debugging Guide

### If Points Are Misaligned:

**Test Pattern:**
```python
# In code, try these translations one by one:

# Option 1: Original
self.t_l2c = np.array([1.878, 0.0, -0.764])

# Option 2: Flip forward/back
self.t_l2c = np.array([-1.878, 0.0, -0.764])

# Option 3: Flip up/down
self.t_l2c = np.array([1.878, 0.0, 0.764])

# Option 4: Flip both
self.t_l2c = np.array([-1.878, 0.0, 0.764])
```

**How to test:**
1. Run `test_fusion.py`
2. Look at `test_projection.jpg`
3. Check if dots align with objects

**Visual signs:**
```
Good alignment:
  Tree → Brown dots
  Grass → Green dots
  Sky → Blue dots (few, distant)

Bad alignment (flip needed):
  Tree → Green dots (wrong!)
  Grass → Brown dots (wrong!)
  All shifted left/right/up/down
```

### If No Points Visible:

**Check point Z distribution:**
```python
# Add this in code to debug:
print(f"LiDAR Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
print(f"Camera Z after transform: [{Z.min():.2f}, {Z.max():.2f}]")
print(f"Points with Z>0: {(Z>0).sum()} / {len(Z)}")
```

**Expected:**
- LiDAR Z: Might be around -10 to +10m (depends on tilt)
- Camera Z: Should have some > 0 (in front)
- If all Z < 0 → Camera facing wrong way!

---

## 📏 FOV Visualization

### 90° FOV means:
```
Camera view (top down):

        Objects here invisible
              |
         45°  |  45°
           \\|//
            \|/
           [C] Camera
```

**At 5m distance:**
- Left edge: 5 × tan(45°) = 5m left
- Right edge: 5 × tan(45°) = 5m right
- Total width visible: 10m

**At 10m distance:**
- Total width: 20m

**In image (640 pixels wide):**
- Each pixel covers: 10m / 640 = 1.56cm at 5m distance
- Each pixel covers: 20m / 640 = 3.1cm at 10m distance

---

## 🎯 Quick Calibration Verification

### Visual Checks:

**1. Distance check:**
```
Pick obvious object (tree, sign, etc.)
LiDAR says: X meters away
Count image pixels from center
Expected: X/focal_length × pixels

Example:
- Tree is 5m away (LiDAR)
- Focal length: 320 pixels
- Should project to: 5/320 = 0.015625... wait that's wrong

Actually: pixel = K × (X/Z, Y/Z)
For tree at [5, 0, 5]:
  pixel = [320×5/5 + 320, ...] = [640, ...]
  → Right edge of image ✓
```

**2. Color check:**
```
Look at test_projection.jpg (RGB-colored version)
- Green grass → Green dots ✓
- Brown tree → Brown dots ✓
- Gray road → Gray dots ✓

If colors don't match → Calibration wrong!
```

**3. Density check:**
```
Objects closer should have:
- More dots (LiDAR resolution)
- Larger on image
- Bluer color (in depth mode)

If far things are blue → Calibration flipped!
```

---

## 📚 Summary

**Coordinate systems:**
- Unreal (left) vs ROS (right) → Flip Y axis
- Both: X=forward, Z=up

**Your setup:**
- Camera 1.878m forward, 0.764m below LiDAR
- Both pointing forward (no rotation)
- 90° FOV, 640×480 resolution

**Troubleshooting:**
- No points → Check Z values, try flips
- Wrong colors → Try coordinate flips
- Outside image → Normal for very close/far objects

**Remember:**
Test script shows you exactly what's happening!
Use it to debug before running main code.

---

Good luck! 🚀
