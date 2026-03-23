"""
Analyze the converted Foggy Driving dataset
"""

import os
import cv2
from pathlib import Path

def analyze_dataset():
    """Analyze the YOLO format dataset"""
    
    base_path = "data/foggy_cityscapes/FC01"
    
    print("\n" + "="*60)
    print("📊 FOGGY DRIVING DATASET ANALYSIS")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    total_images = 0
    total_objects = 0
    
    for split in splits:
        img_dir = f"{base_path}/images/{split}"
        lbl_dir = f"{base_path}/labels/{split}"
        
        if not os.path.exists(img_dir):
            continue
        
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"\n{split.upper()}: No data")
            continue
        
        num_images = len(images)
        total_images += num_images
        
        print(f"\n📁 {split.upper()} SET:")
        print(f"   Images: {num_images}")
        
        total_split_objects = 0
        
        for img_file in images:
            lbl_file = Path(img_file).stem + '.txt'
            lbl_path = os.path.join(lbl_dir, lbl_file)
            
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    objects = len([l for l in f.readlines() if l.strip()])
                    total_split_objects += objects
            
            # Get image info
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
        
        total_objects += total_split_objects
        avg_objects = total_split_objects / num_images if num_images > 0 else 0
        print(f"   Total Objects: {total_split_objects}")
        print(f"   Avg Objects/Image: {avg_objects:.2f}")
        print(f"   Image Size: {w}×{h}")
    
    print(f"\n{'='*60}")
    print(f"📈 SUMMARY:")
    print(f"   Total Images: {total_images}")
    print(f"   Total Objects: {total_objects}")
    print(f"   Avg Objects/Image: {total_objects/total_images:.2f}")
    print(f"   Classes: 1 (Pedestrian/Objects)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    analyze_dataset()
