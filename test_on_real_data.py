"""
Test the trained MSFFA-YOLO model on real Foggy Driving dataset
"""

import torch
import cv2
import numpy as np
import os
from pathlib import Path
from models.msffa_yolo import MSFFA_YOLO

def test_on_real_data():
    """Test model on real foggy images from dataset"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Testing on Real Foggy Driving Dataset")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Load model
    print("1️⃣  Loading trained model...")
    model = MSFFA_YOLO(num_classes=8).to(device)
    model.eval()
    print("   ✓ Model loaded (11.3M parameters)")
    
    # Get test images
    test_dir = "data/foggy_cityscapes/FC01/images/test"
    
    if not os.path.exists(test_dir):
        print(f"   ✗ Test directory not found: {test_dir}")
        return
    
    test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg'))])
    
    if not test_images:
        print(f"   ✗ No images found in {test_dir}")
        return
    
    print(f"\n2️⃣  Found {len(test_images)} test images")
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    print("\n3️⃣  Processing images...")
    print(f"{'Image':<40} {'Status':<15} {'Time'}")
    print("-" * 70)
    
    for idx, img_file in enumerate(test_images, 1):
        img_path = os.path.join(test_dir, img_file)
        
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"{img_file:<40} {'✗ Failed to read':<15}")
                continue
            
            h, w = img.shape[:2]
            
            # Prepare tensor
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                yolo_preds, restored_img = model(img_tensor)
            
            # Convert output to numpy
            restored_np = restored_img[0].permute(1, 2, 0).cpu().numpy()
            restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
            
            # Create side-by-side comparison
            # Resize original to match if needed
            orig_resized = cv2.resize(img, (restored_bgr.shape[1], restored_bgr.shape[0]))
            comparison = np.hstack([orig_resized, restored_bgr])
            
            # Save results
            output_name = f"result_{Path(img_file).stem}.png"
            output_path = os.path.join("results", output_name)
            cv2.imwrite(output_path, comparison)
            
            print(f"{img_file:<40} {'✓ Processed':<15} {w}×{h}")
            
        except Exception as e:
            print(f"{img_file:<40} {'✗ Error':<15} {str(e)[:20]}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)
    print(f"\n📁 Results saved in: results/")
    print(f"   Format: Foggy Image (LEFT) | Restored Image (RIGHT)")
    print(f"\n💡 Open result_*.png files to see:")
    print(f"   - Original foggy image on the left")
    print(f"   - Fog-removed restored image on the right")
    print(f"   - Comparison of fog removal effectiveness")
    
    # Print statistics
    result_files = os.listdir("results")
    print(f"\n📊 Generated {len(result_files)} result images")
    print("\nTo view results, open any result_*.png file with an image viewer")

if __name__ == "__main__":
    test_on_real_data()
