"""Extract point clouds from ROS2 bag and check for labels"""
import sqlite3
import numpy as np
from pathlib import Path
import argparse
import struct

def extract_pointclouds(bag_path, output_dir, topic_name='/lidar/points2', max_clouds=20):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    bag_file = Path(bag_path)
    if bag_file.suffix != '.db3':
        db_files = list(bag_file.glob('*.db3'))
        bag_file = db_files[0]
    
    print("="*70)
    print(f"Extracting from: {bag_file}")
    print(f"Topic: {topic_name}")
    print(f"Max clouds: {max_clouds}")
    print("="*70)
    
    conn = sqlite3.connect(str(bag_file))
    cursor = conn.cursor()
    
    # Get topic ID
    cursor.execute("SELECT id FROM topics WHERE name=?", (topic_name,))
    result = cursor.fetchone()
    
    if not result:
        print(f"\nERROR: Topic '{topic_name}' not found!")
        print("Run inspect_rosbag.py to see available topics")
        conn.close()
        return
    
    topic_id = result[0]
    
    # Get messages
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,))
    messages = cursor.fetchall()
    
    print(f"\nFound {len(messages)} point cloud messages")
    print(f"Extracting {min(max_clouds, len(messages))} samples...\n")
    
    all_fields_found = set()
    all_labels_found = set()
    
    for idx in range(min(max_clouds, len(messages))):
        timestamp, data = messages[idx * (len(messages) // max_clouds)]  # Sample evenly
        
        try:
            # Parse PointCloud2 header
            offset = 4  # Skip CDR header
            
            # Skip std_msgs/Header
            offset += 8  # timestamp
            frame_id_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4 + frame_id_len
            while offset % 4 != 0:
                offset += 1
            
            # PointCloud2 fields
            height = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            width = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            num_points = height * width
            
            # Read fields
            fields_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            
            fields = []
            print(f"\nCloud {idx+1}: {num_points} points")
            print("  Fields found:")
            
            for _ in range(fields_len):
                name_len = struct.unpack_from('<I', data, offset)[0]
                offset += 4
                name = data[offset:offset+name_len].decode('utf-8').rstrip('\x00')
                offset += name_len
                while offset % 4 != 0:
                    offset += 1
                
                field_offset = struct.unpack_from('<I', data, offset)[0]
                offset += 4
                datatype = struct.unpack_from('B', data, offset)[0]
                offset += 1
                while offset % 4 != 0:
                    offset += 1
                count = struct.unpack_from('<I', data, offset)[0]
                offset += 4
                
                fields.append({'name': name, 'offset': field_offset, 'datatype': datatype})
                print(f"    - {name} (offset: {field_offset}, type: {datatype})")
                all_fields_found.add(name)
            
            # Check for label fields
            has_labels = any('label' in f['name'].lower() or 
                           'semantic' in f['name'].lower() or
                           'class' in f['name'].lower() 
                           for f in fields)
            
            if has_labels:
                print(" HAS LABELS!")
                all_labels_found.add(idx)
            else:
                print("No label fields found")
            
        except Exception as e:
            print(f"  Error parsing cloud {idx}: {e}")
    
    conn.close()
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Extracted: {max_clouds} point clouds")
    print(f"\nAll fields found across all clouds:")
    for field in sorted(all_fields_found):
        print(f"  - {field}")
    
    if all_labels_found:
        print(f"\nLABELS FOUND IN {len(all_labels_found)} CLOUDS! ✓✓✓")
        print("Your data HAS semantic labels!")
    else:
        print("\nNO LABELS FOUND ✗✗✗")
        print("Your point clouds don't have semantic labels")
        print("You'll need to use camera images or UE export instead")
    
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and inspect point clouds')
    parser.add_argument('bag_path', help='Path to ROS2 bag')
    parser.add_argument('--output', default='../data/test_pointclouds', help='Output directory')
    parser.add_argument('--topic', default='/lidar/points2', help='Point cloud topic')
    parser.add_argument('--max-clouds', type=int, default=50, help='Number of clouds to check')
    
    args = parser.parse_args()
    extract_pointclouds(args.bag_path, args.output, args.topic, args.max_clouds)
