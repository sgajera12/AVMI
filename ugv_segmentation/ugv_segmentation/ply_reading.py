# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:53:42 2025

@author: fhs95
"""

import open3d as o3d
import numpy as np

PLY = "pass_both_projection.ply"

# 读取 ply
pcd = o3d.io.read_point_cloud(PLY)
print(pcd)

# 显示
o3d.visualization.draw_geometries(
    [pcd],
    window_name="Pass_both Projection (PLY)",
    width=1280,
    height=800,
    point_show_normal=False
)
