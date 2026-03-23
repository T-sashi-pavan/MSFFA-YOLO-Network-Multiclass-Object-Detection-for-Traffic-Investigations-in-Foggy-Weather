"""
Convert Foggy Driving dataset from bboxGt format to YOLO format
bboxGt format: class_id x1 y1 x2 y2 (pixel coordinates)
YOLO format: class_id x_center y_center width height (normalized 0-1)
"""

import os
import shutil
import cv2
from pathlib import Path
import json

# Class mapping (Single class: pedestrian = 0)
CLASS_MAP = {
    'pedestrian': 0,
    'public': 0  # Both folders represent pedestrians/objects
}

def get_image_dimensions(img_path):
    """Get image width and height"""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    height, width = img.shape[:2]
    return width, height

def bbox_pixel_to_yolo(x1, y1, x2, y2, width, height):
    """
    Convert pixel coordinates to YOLO format
    Input: x1, y1, x2, y2 (pixel coordinates)
    Output: x_center, y_center, w, h (normalized 0-1)
    """
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    
    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    
    return x_center, y_center, w, h

def convert_dataset():
    """Main conversion function"""
    
    base_dir = "Foggy_Driving"
    output_base = "data/foggy_cityscapes"
    
    # Create output directories
    os.makedirs(f"{output_base}/FC01/images/train", exist_ok=True)
    os.makedirs(f"{output_base}/FC01/images/val", exist_ok=True)
    os.makedirs(f"{output_base}/FC01/images/test", exist_ok=True)
    os.makedirs(f"{output_base}/FC01/labels/train", exist_ok=True)
    os.makedirs(f"{output_base}/FC01/labels/val", exist_ok=True)
    os.makedirs(f"{output_base}/FC01/labels/test", exist_ok=True)
    
    print("Created directory structure:")
    print(f"  {output_base}/FC01/images/")
    print(f"  {output_base}/FC01/labels/")
    
    # Process each category
    categories = ['pedestrian', 'public']
    split_type = 'test'  # We only have test data
    
    total_images = 0
    total_converted = 0
    total_skipped = 0
    
    for category in categories:
        img_dir = f"{base_dir}/leftImg8bit/{split_type}/{category}"
        bbox_dir = f"{base_dir}/bboxGt/{split_type}/{category}"
        
        if not os.path.exists(img_dir):
            print(f"⚠️  {img_dir} not found, skipping...")
            continue
        
        print(f"\n📁 Processing: {category}")
        
        # Get all images in category
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        for img_file in img_files:
            total_images += 1
            
            img_path = os.path.join(img_dir, img_file)
            bbox_file = img_file.replace('_leftImg8bit.png', '.txt')
            bbox_path = os.path.join(bbox_dir, bbox_file)
            
            # Get image dimensions
            width, height = get_image_dimensions(img_path)
            if width is None:
                print(f"   ⚠️  Failed to read: {img_file}")
                total_skipped += 1
                continue
            
            # Copy image to test folder
            output_img_dir = f"{output_base}/FC01/images/test"
            output_img_path = os.path.join(output_img_dir, img_file)
            
            try:
                shutil.copy2(img_path, output_img_path)
            except Exception as e:
                print(f"   ✗ Error copying {img_file}: {e}")
                total_skipped += 1
                continue
            
            # Convert labels
            output_label_dir = f"{output_base}/FC01/labels/test"
            output_label_file = img_file.replace('_leftImg8bit.png', '.txt')
            output_label_path = os.path.join(output_label_dir, output_label_file)
            
            if os.path.exists(bbox_path):
                try:
                    with open(bbox_path, 'r') as f:
                        lines = f.readlines()
                    
                    yolo_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x1, y1, x2, y2 = map(float, parts[1:])
                            
                            x_center, y_center, w, h = bbox_pixel_to_yolo(
                                x1, y1, x2, y2, width, height
                            )
                            
                            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    
                    # Write YOLO format labels
                    with open(output_label_path, 'w') as f:
                        f.writelines(yolo_lines)
                    
                    total_converted += 1
                    print(f"   ✓ {img_file} ({len(yolo_lines)} objects)")
                    
                except Exception as e:
                    print(f"   ✗ Error converting {bbox_file}: {e}")
                    total_skipped += 1
            else:
                print(f"   ⚠️  No labels for {img_file}")
                total_skipped += 1
    
    print(f"\n" + "="*60)
    print(f"Conversion Summary:")
    print(f"  Total images: {total_images}")
    print(f"  Successfully converted: {total_converted}")
    print(f"  Skipped/Failed: {total_skipped}")
    print(f"  Output location: {output_base}/FC01/")
    print(f"="*60)

if __name__ == "__main__":
    print("🔄 Converting Foggy Driving Dataset to YOLO Format")
    print("="*60)
    convert_dataset()
    print("\n✓ Conversion complete!")
