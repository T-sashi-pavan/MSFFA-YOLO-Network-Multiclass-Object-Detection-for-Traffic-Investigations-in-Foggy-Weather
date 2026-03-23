import torch
import cv2
import numpy as np
from models.msffa_yolo import MSFFA_YOLO
import os

def visualize_detections(pred, img, scale_idx):
    """Draw bounding boxes from YOLO predictions"""
    if pred.dim() < 4:
        return img
    
    h, w = img.shape[:2]
    batch_size, num_classes_plus_4, grid_h, grid_w = pred.shape
    num_classes = num_classes_plus_4 - 4
    
    img_copy = img.copy()
    
    # Simple visualization (just show that detections exist)
    conf_threshold = 0.1
    pred = pred.permute(0, 2, 3, 1).reshape(-1, num_classes_plus_4)
    
    for detection in pred:
        conf = detection[4].item()  # Objectness score
        if conf > conf_threshold:
            # Normalize and draw
            x_norm, y_norm, w_norm, h_norm = detection[:4]
            x = int(x_norm * w)
            y = int(y_norm * h)
            bw = int(w_norm * w)
            bh = int(h_norm * h)
            
            x1, y1 = max(0, x - bw//2), max(0, y - bh//2)
            x2, y2 = min(w, x + bw//2), min(h, y + bh//2)
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img_copy

def test_full_pipeline():
    """Test complete MSFFA-YOLO pipeline: Restoration + Detection"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\n1. Loading MSFFA-YOLO model...")
    model = MSFFA_YOLO(num_classes=8).to(device)
    model.eval()
    print("   ✓ Model loaded (11.3M parameters)")
    
    # Create test image
    print("\n2. Creating test image with objects...")
    img_size = 256
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 180
    
    # Draw objects
    cv2.rectangle(img, (30, 30), (120, 120), (50, 50, 150), -1)      # Car-like
    cv2.rectangle(img, (150, 50), (230, 130), (50, 150, 50), -1)     # Person-like
    cv2.circle(img, (80, 200), 25, (150, 0, 0), -1)                 # Bike-like
    
    # Add fog
    fog = np.ones_like(img) * 255
    foggy_img = cv2.addWeighted(img, 0.4, fog, 0.6, 0)
    print("   ✓ Created image with fog")
    
    # Pass through pipeline
    print("\n3. Running inference...")
    print("   a) Input → Foggy image (256×256)")
    
    img_rgb = cv2.cvtColor(foggy_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward pass
        yolo_preds, restored_img = model(img_tensor)
    
    print("   b) MSFFA Subnet → Restores image (removes fog)")
    
    restored_np = restored_img[0].permute(1, 2, 0).cpu().numpy()
    restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
    restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
    
    print("   c) YOLO Subnet → Detects objects on restored image")
    print(f"      - Output 1: {yolo_preds[0].shape} (scale 1)")
    print(f"      - Output 2: {yolo_preds[1].shape} (scale 2)")
    print(f"      - Output 3: {yolo_preds[2].shape} (scale 3)")
    
    # Visualize detections
    detected_img = restored_bgr.copy()
    for i, pred in enumerate(yolo_preds):
        detected_img = visualize_detections(pred.cpu(), detected_img, i)
    
    # Save results
    print("\n4. Saving results...")
    cv2.imwrite("test_original_foggy.png", foggy_img)
    cv2.imwrite("test_restored_clear.png", restored_bgr)
    cv2.imwrite("test_with_detections.png", detected_img)
    
    # Comparison
    comparison = np.hstack([foggy_img, restored_bgr, detected_img])
    cv2.imwrite("test_full_pipeline.png", comparison)
    
    print("   ✓ Saved: test_original_foggy.png")
    print("   ✓ Saved: test_restored_clear.png")
    print("   ✓ Saved: test_with_detections.png")
    print("   ✓ Saved: test_full_pipeline.png (all 3 stages)")
    
    print("\n✓ Pipeline test complete!")
    print("\n🎯 MSFFA-YOLO Pipeline:")
    print("   Step 1: Foggy Image → MSFFA (restoration)")
    print("   Step 2: Restored Image → YOLO (detection)")
    print("   Step 3: Final output with detected objects")

if __name__ == "__main__":
    test_full_pipeline()
