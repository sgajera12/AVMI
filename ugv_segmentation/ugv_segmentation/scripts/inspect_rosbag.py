"""Inspect ROS2 bag to see what's inside"""
import sqlite3
from pathlib import Path
import argparse

def inspect_bag(bag_path):
    bag_file = Path(bag_path)
    if bag_file.suffix != '.db3':
        db_files = list(bag_file.glob('*.db3'))
        if not db_files:
            print(f"ERROR: No .db3 file found")
            return
        bag_file = db_files[0]
    
    print("="*70)
    print(f"INSPECTING: {bag_file}")
    print("="*70)
    
    conn = sqlite3.connect(str(bag_file))
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()
    
    print(f"\nFOUND {len(topics)} TOPICS:\n")
    
    for topic_id, topic_name, topic_type in topics:
        cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,))
        msg_count = cursor.fetchone()[0]
        
        print(f"Topic: {topic_name}")
        print(f"  Type: {topic_type}")
        print(f"  Messages: {msg_count}\n")
    
    print("\n" + "="*70)
    print("POINT CLOUD TOPICS:")
    for topic_id, topic_name, topic_type in topics:
        if 'point' in topic_name.lower() or 'lidar' in topic_name.lower():
            print(f"  ✓ {topic_name}")
    
    print("\nIMAGE TOPICS:")
    for topic_id, topic_name, topic_type in topics:
        if 'image' in topic_name.lower() or 'camera' in topic_name.lower():
            print(f"  ✓ {topic_name}")
    
    conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_path', help='Path to ROS2 bag')
    args = parser.parse_args()
    inspect_bag(args.bag_path)
