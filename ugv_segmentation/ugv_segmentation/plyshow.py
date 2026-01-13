import subprocess
import sys

def fix_numpy():
    print("Fixing NumPy compatibility...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "--break-system-packages"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "open3d", "--break-system-packages"], check=True)
    print("Fixed! Restarting script...\n")
    subprocess.run([sys.executable, __file__], check=True)
    sys.exit(0)

try:
    import open3d as o3d
    import numpy as np
except AttributeError:
    fix_numpy()

PLY = "pass_both_projection.ply"
pcd = o3d.io.read_point_cloud(PLY)
print(pcd)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Pass_both Projection (PLY)",
    width=1200,
    height=800,
    point_show_normal=False
)