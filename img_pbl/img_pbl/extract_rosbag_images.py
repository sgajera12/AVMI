"""
SIMPLE ROS2 BAG IMAGE EXTRACTOR

Works with any ROS2 bag format, no external ROS dependencies needed!
Uses only sqlite3 (built-in) and opencv
"""

import sqlite3
import cv2
import numpy as np
from pathlib import Path
import argparse
import struct


def extract_images_simple(bag_path, output_dir, topic_name='/camera/left/image_raw', 
                          sample_every=1, max_images=None):
    """
    Extract images from ROS2 bag using direct sqlite3 access
    No rosbags library needed!
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # ROS2 bags are sqlite3 databases
    bag_file = Path(bag_path)
    if bag_file.suffix != '.db3':
        # If directory given, find the .db3 file
        db_files = list(bag_file.glob('*.db3'))
        if not db_files:
            print(f"ERROR: No .db3 file found in {bag_file}")
            return
        bag_file = db_files[0]
    
    print(f"Reading bag: {bag_file}")
    print(f"Topic: {topic_name}")
    print(f"Sampling every {sample_every} frames")
    
    try:
        # Connect to sqlite database
        conn = sqlite3.connect(str(bag_file))
        cursor = conn.cursor()
        
        # Get topic_id for our topic
        cursor.execute("SELECT id FROM topics WHERE name=?", (topic_name,))
        result = cursor.fetchone()
        topic_type = cursor.fetchone()[0]
        print(f"Found topic ID: {topic_id}, type: {topic_type}")
        
        if not result:
            print(f"\nERROR: Topic '{topic_name}' not found!")
            print("\nAvailable topics:")
            cursor.execute("SELECT name FROM topics")
            for (name,) in cursor.fetchall():
                print(f"  - {name}")
            conn.close()
            return
        
        topic_id = result[0]
        print(f"Found topic ID: {topic_id}")


        # Get all messages for this topic
        cursor.execute("""
            SELECT timestamp, data 
            FROM messages 
            WHERE topic_id=? 
            ORDER BY timestamp
        """, (topic_id,))
        
        messages = cursor.fetchall()
        print(f"Found {len(messages)} messages")
        
        saved_count = 0
        
        for idx, (timestamp, data) in enumerate(messages):
            if max_images and saved_count >= max_images:
                break
            
            if idx % sample_every != 0:
                continue
            
            try:
                # Parse ROS2 Image message
                img = parse_ros2_image(data)
                
                if img is not None:
                    # Save image
                    filename = f"frame_{saved_count:06d}_{timestamp}.jpg"
                    filepath = output_dir / filename
                    cv2.imwrite(str(filepath), img)
                    saved_count += 1
                    
                    if saved_count % 10 == 0:
                        print(f"Extracted {saved_count} images...")
            
            except Exception as e:
                print(f"Error while parsing image message: {e}")
                import traceback
                traceback.print_exc()
                return None

        
        conn.close()
        print(f"\n✓ Done! Extracted {saved_count} images to {output_dir}")
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def parse_ros2_compressed_image(data):
    """
    Parse ROS2 sensor_msgs/msg/CompressedImage
    header: std_msgs/Header
    string format
    uint8[] data (JPEG/PNG)
    """
    offset = 4  # skip CDR header

    # --- skip stamp.sec + stamp.nanosec (8 bytes total) ---
    offset += 8

    # --- frame_id (string: length + data) ---
    frame_id_len = struct.unpack_from('<I', data, offset)[0]
    offset += 4 + frame_id_len

    # align to 4
    while offset % 4 != 0:
        offset += 1

    # --- format string ---
    fmt_len = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    fmt_str = data[offset:offset+fmt_len].decode('utf-8')
    offset += fmt_len

    # align to 4
    while offset % 4 != 0:
        offset += 1

    # --- image data (uint8[]) ---
    data_len = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    img_data = data[offset:offset+data_len]

    # Decode with OpenCV
    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR image

    if img is None:
        print(f"Warning: cv2.imdecode failed (format={fmt_str})")
    return img


def parse_ros2_image(data):
    """
    Simple ROS2 Image message parser
    Extracts image from CDR serialized data
    """
    try:
        # ROS2 uses CDR serialization
        # Skip CDR header (4 bytes)
        offset = 4
        
        # Skip std_msgs/Header
        # sequence (4 bytes) - skipped in ROS2
        # stamp sec (4 bytes)
        # stamp nanosec (4 bytes)
        offset += 8
        
        # frame_id (string: 4 bytes length + string)
        frame_id_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4 + frame_id_len
        
        # Align to 4-byte boundary
        while offset % 4 != 0:
            offset += 1
        
        # Image data
        height = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        width = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # encoding (string)
        encoding_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        encoding = data[offset:offset+encoding_len].decode('utf-8')
        offset += encoding_len
        
        # Align to 4-byte boundary
        while offset % 4 != 0:
            offset += 1
        
        # is_bigendian, step
        is_bigendian = struct.unpack_from('B', data, offset)[0]
        offset += 1
        
        # Align to 4-byte boundary
        while offset % 4 != 0:
            offset += 1
            
        step = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # data (array: 4 bytes length + data)
        data_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        img_data = data[offset:offset+data_len]
        
        # Convert to numpy array
        if 'bgr8' in encoding.lower():
            img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
        elif 'rgb8' in encoding.lower():
            img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif 'mono8' in encoding.lower():
            img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
        else:
            print(f"Warning: Unsupported encoding '{encoding}'")
            return None
        
        return img
    
    except Exception as e:
        # If parsing fails, return None
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract images from ROS2 bag (simple method)')
    parser.add_argument('bag_path', type=str, help='Path to ROS2 bag (.db3 file)')
    parser.add_argument('--output', type=str, default='./extracted_frames',
                       help='Output directory for images')
    parser.add_argument('--topic', type=str, default='/camera/left/image_raw',
                       help='Camera topic name')
    parser.add_argument('--sample-every', type=int, default=10,
                       help='Extract every Nth frame')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum images to extract')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SIMPLE ROS2 BAG IMAGE EXTRACTOR")
    print("="*60)
    
    extract_images_simple(
        bag_path=args.bag_path,
        output_dir=args.output,
        topic_name=args.topic,
        sample_every=args.sample_every,
        max_images=args.max_images
    )