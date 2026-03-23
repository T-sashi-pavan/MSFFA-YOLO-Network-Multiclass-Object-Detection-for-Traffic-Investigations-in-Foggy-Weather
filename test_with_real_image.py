"""
Test with real foggy image to demonstrate the issue
"""
import torch
import cv2
import numpy as np
from PIL import Image
import os
from models.msffa_yolo import MSFFA_YOLO

device = "cpu"
model = MSFFA_YOLO(num_classes=8).to(device)
model.eval()

print("=" * 70)
print("TESTING WITH REAL FOGGY IMAGE")
print("=" * 70)

# Look for any foggy image in the dataset
foggy_dir = "Foggy_Driving/leftImg8bit"
images = []

if os.path.exists(foggy_dir):
    # Recursively find all images
    for root, dirs, files in os.walk(foggy_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                images.append(os.path.join(root, file))
                if len(images) >= 1:  # Just get first one
                    break

if images:
    test_image_path = images[0]
    print(f"\n✓ Found test image: {test_image_path}")
    
    # Load image
    image_bgr = cv2.imread(test_image_path)
    if image_bgr is None:
        print(f"ERROR: Could not load image from {test_image_path}")
    else:
        h, w = image_bgr.shape[:2]
        print(f"  Image size: {w}x{h}")
        
        # Prepare for model
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        
        print(f"\nInput tensor shape: {img_tensor.shape}")
        print(f"Input range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
        
        # Run model
        with torch.no_grad():
            yolo_preds, restored_img = model(img_tensor)
        
        print(f"\nModel output shape: {restored_img.shape}")
        print(f"Model output range: [{restored_img.min():.4f}, {restored_img.max():.4f}]")
        
        # Convert to uint8
        restored_np = restored_img[0].permute(1, 2, 0).cpu().numpy()
        
        if restored_np.min() < 0:
            restored_np = (restored_np + 1) / 2.0
            restored_np = (restored_np * 255).astype(np.uint8)
        else:
            restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
        
        print(f"\nConverted output shape: {restored_np.shape}")
        print(f"Converted output range: [{restored_np.min()}, {restored_np.max()}]")
        
        # Save comparison
        comparison = np.hstack([image_bgr, cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)])
        cv2.imwrite('untrained_model_output.jpg', comparison)
        print(f"\n✓ Saved comparison to: untrained_model_output.jpg")
        
        # Show statistics
        print(f"\nStatistics:")
        print(f"  Original brightness: {image_bgr.astype(np.float32).mean():.1f}")
        print(f"  Restored brightness: {restored_np.astype(np.float32).mean():.1f}")
        print(f"  Original contrast: {image_bgr.astype(np.float32).std():.1f}")
        print(f"  Restored contrast: {restored_np.astype(np.float32).std():.1f}")

else:
    print(f"\n⚠ No images found in {foggy_dir}")
    print("The dataset needs to be loaded first!")

print("\n" + "=" * 70)
print("SOLUTION: The model needs to be TRAINED to produce meaningful output!")
print("=" * 70)
