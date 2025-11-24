from pathlib import Path
import open3d as o3d
frame_id = 200
ply_file = f"ply/colored_lidar_{frame_id:06d}.ply"

pcd = o3d.io.read_point_cloud(ply_file)
o3d.visualization.draw_geometries([pcd])