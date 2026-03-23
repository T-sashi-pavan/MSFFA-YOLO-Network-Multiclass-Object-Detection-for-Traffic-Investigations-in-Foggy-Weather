import torch
import cv2
import numpy as np
from models.msffa_yolo import MSFFA_YOLO
import os

def test_restoration():
    """Test the fog removal capability of MSFFA-YOLO model"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\n1. Loading model...")
    model = MSFFA_YOLO(num_classes=8).to(device)
    model.eval()
    print("   ✓ Model loaded")
    
    # Create synthetic foggy image (smaller for faster processing)
    print("\n2. Creating synthetic foggy image...")
    img_size = 256  # Smaller for faster processing
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200  # Gray/foggy background
    
    # Draw some objects
    cv2.rectangle(img, (50, 50), (150, 150), (100, 0, 0), -1)    # Dark rectangle
    cv2.rectangle(img, (170, 170), (230, 230), (0, 100, 0), -1)  # Green rectangle
    cv2.circle(img, (60, 200), 30, (0, 0, 100), -1)              # Blue circle
    
    # Add fog effect (blend with white)
    fog = np.ones_like(img) * 255
    alpha = 0.6
    foggy_img = cv2.addWeighted(img, 1-alpha, fog, alpha, 0)
    
    cv2.imwrite("test_original_foggy.png", foggy_img)
    print("   ✓ Created foggy image (256x256)")
    
    # Prepare image for model
    print("\n3. Processing through MSFFA Restoration Subnet...")
    img_rgb = cv2.cvtColor(foggy_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    print(f"   Input shape: {img_tensor.shape}")
    
    with torch.no_grad():
        # Get restoration output
        yolo_preds, restored_img = model(img_tensor)
    
    print("   ✓ Restoration complete")
    print(f"   Output shape: {restored_img.shape}")
    
    # Convert restored image to numpy
    restored_np = restored_img[0].permute(1, 2, 0).cpu().numpy()
    restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
    restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
    
    # Save restored
    cv2.imwrite("test_restored_clear.png", restored_bgr)
    print("   ✓ Saved: test_restored_clear.png")
    
    # Create comparison image
    comparison = np.hstack([foggy_img, restored_bgr])
    cv2.imwrite("test_comparison.png", comparison)
    print("   ✓ Saved: test_comparison.png (side-by-side)")
    
    # Statistics
    print("\n4. Image Quality Comparison:")
    print(f"   {'Metric':<25} {'Foggy':<12} {'Restored':<12}")
    print(f"   {'-'*50}")
    print(f"   {'Mean Brightness':<25} {foggy_img.mean():>11.2f} {restored_bgr.mean():>11.2f}")
    print(f"   {'Std Deviation':<25} {foggy_img.std():>11.2f} {restored_bgr.std():>11.2f}")
    print(f"   {'Min Value':<25} {foggy_img.min():>11.2f} {restored_bgr.min():>11.2f}")
    print(f"   {'Max Value':<25} {foggy_img.max():>11.2f} {restored_bgr.max():>11.2f}")
    
    print("\n✓ Testing complete!")
    print("\n📊 Results saved in project root:")
    print("   - test_original_foggy.png")
    print("   - test_restored_clear.png")
    print("   - test_comparison.png")
    print("\nOpen test_comparison.png to see side-by-side comparison!")

if __name__ == "__main__":
    test_restoration()
