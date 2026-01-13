#!/bin/bash
# ALL-IN-ONE IMAGE EXTRACTOR
# Extracts images with correct QoS settings

BAG_PATH=$1
OUTPUT_DIR=${2:-"./images_extracted"}
TOPIC=${3:-"/camera/left/image_raw"}
SAMPLE_RATE=${4:-50}
MAX_IMAGES=${5:-50}

if [ -z "$BAG_PATH" ]; then
    echo "Usage: ./extract_images.sh <bag_path> [output_dir] [topic] [sample_rate] [max_images]"
    echo ""
    echo "Example:"
    echo "  ./extract_images.sh /path/to/bag.db3 ./images /camera/left/image_raw 50 50"
    exit 1
fi

echo "============================================"
echo "IMAGE EXTRACTION WITH QoS"
echo "============================================"
echo "Bag: $BAG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Topic: $TOPIC"
echo "Sample: every $SAMPLE_RATE frames"
echo "Max: $MAX_IMAGES images"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create the extractor script
cat > /tmp/extract_qos.py << 'EOFPYTHON'
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
from pathlib import Path

class ImageExtractor(Node):
    def __init__(self, topic, output_dir, sample_rate, max_images):
        super().__init__('image_extractor')
        self.bridge = CvBridge()
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.max_images = max_images
        self.count = 0
        self.saved = 0
        
        # BEST_EFFORT QoS (like RViz)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.subscription = self.create_subscription(Image, topic, self.callback, qos)
        self.get_logger().info(f'Ready to receive images from {topic}')
    
    def callback(self, msg):
        self.count += 1
        if self.count % self.sample_rate != 0:
            return
        if self.saved >= self.max_images:
            self.get_logger().info('Done!')
            rclpy.shutdown()
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            filename = self.output_dir / f'frame_{self.saved:04d}.jpg'
            cv2.imwrite(str(filename), cv_image)
            self.saved += 1
            if self.saved % 10 == 0:
                self.get_logger().info(f'Saved {self.saved}/{self.max_images}')
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main():
    rclpy.init()
    node = ImageExtractor(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    try:
        rclpy.spin(node)
    except:
        pass
    print(f'\n✓ Extracted {node.saved} images to {sys.argv[2]}')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
EOFPYTHON

chmod +x /tmp/extract_qos.py

# Start the extractor in background
echo ""
echo "Starting image extractor..."
python3 /tmp/extract_qos.py "$TOPIC" "$OUTPUT_DIR" "$SAMPLE_RATE" "$MAX_IMAGES" &
EXTRACTOR_PID=$!

# Wait a moment for it to start
sleep 2

# Play the bag
echo "Playing bag..."
ros2 bag play "$BAG_PATH" --rate 5.0 > /dev/null 2>&1

# Wait for extractor to finish
wait $EXTRACTOR_PID

# Clean up
rm /tmp/extract_qos.py

# Count results
EXTRACTED=$(ls "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)

if [ $EXTRACTED -gt 0 ]; then
    echo "Success"
else
    echo "FAILED!"
fi
