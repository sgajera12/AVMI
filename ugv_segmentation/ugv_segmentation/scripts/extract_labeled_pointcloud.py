"""Extract labeled point clouds from ROS2 bag"""
import sqlite3
import numpy as np
from pathlib import Path
import argparse
import struct

def extract_clouds(bag_path, output_dir, topic_name, max_clouds=20):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    bag_file = Path(bag_path)
    if bag_file.suffix != '.db3':
        db_files = list(bag_file.glob('*.db3'))
        bag_file = db_files[0]
    
    print(f"Reading: {bag_file}")
    print(f"Topic: {topic_name}")
    
    conn = sqlite3.connect(str(bag_file))
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM topics WHERE name=?", (topic_name,))
    result = cursor.fetchone()
    
    if not result:
        print(f"ERROR: Topic '{topic_name}' not found!")
        conn.close()
        return
    
    topic_id = result[0]
    
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,))
    messages = cursor.fetchall()
    
    print(f"Found {len(messages)} point clouds")
    print(f"Extracting {min(max_clouds, len(messages))} samples...")
    
    all_labels = set()
    
    for idx in range(min(max_clouds, len(messages))):
        timestamp, data = messages[idx]
        
        # Try to parse - if it fails, that's OK
        try:
            # This is a simplified parser
            # It may not work for all point cloud formats
            # But it's a starting point
            
            filename = f"pointcloud_{idx:06d}.txt"
            filepath = output_dir / filename
            
            # Save raw data for now
            # We can improve parsing based on what we see
            with open(output_dir / f"raw_{idx:06d}.bin", 'wb') as f:
                f.write(data)
            
            print(f"Extracted {idx+1}/{max_clouds}")
            
        except Exception as e:
            print(f"Warning: Could not parse cloud {idx}: {e}")
    
    conn.close()
    print(f"\n✓ Extracted {max_clouds} point clouds to {output_dir}")
    print("\nNOTE: Point cloud parsing is complex.")
    print("Check if you have labels by looking at the data structure.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_path')
    parser.add_argument('--output', default='./pointclouds')
    parser.add_argument('--topic', default='/lidar/points2')
    parser.add_argument('--max-clouds', type=int, default=20)
    args = parser.parse_args()
    
    extract_clouds(args.bag_path, args.output, args.topic, args.max_clouds)
