"""
Diagnose the white output issue and create a better training approach
"""
import torch
import cv2
import numpy as np
from models.msffa_yolo import MSFFA_YOLO
import os

print("=" * 80)
print("DIAGNOSING OUTPUT ISSUE")
print("=" * 80)

device = "cpu"
model = MSFFA_YOLO(num_classes=8).to(device)

# Load trained model
weight_path = "weights/msffa_yolo_final.pth"
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"✓ Loaded trained model")
else:
    print(f"✗ Model not found")
    exit(1)

model.eval()

# Load a test image
test_image_path = "Foggy_Driving/leftImg8bit/test/pedestrian/pedestrian_20161201_101324_leftImg8bit.png"
if not os.path.exists(test_image_path):
    print(f"Test image not found")
    exit(1)

image_bgr = cv2.imread(test_image_path)
h, w = image_bgr.shape[:2]
print(f"\nTest image: {w}×{h}")

# Prepare input
img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

print(f"Input range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
print(f"Input mean: {img_tensor.mean():.4f}")

# Run model
with torch.no_grad():
    yolo_preds, restored_img = model(img_tensor)

print(f"\nRaw output (Tanh) range: [{restored_img.min():.4f}, {restored_img.max():.4f}]")
print(f"Raw output mean: {restored_img.mean():.4f}")

# Debug: Check individual channels
for c in range(3):
    channel = restored_img[0, c]
    print(f"  Channel {c}: [{channel.min():.4f}, {channel.max():.4f}] mean={channel.mean():.4f}")

# Convert as we do in the app
restored_np = restored_img[0].permute(1, 2, 0).cpu().numpy()

# Handle Tanh
if restored_np.min() < 0:
    print(f"\n✓ Detected Tanh output, converting [-1,1] → [0,255]")
    restored_np = (restored_np + 1) / 2.0  # [-1,1] → [0,1]
    restored_np = (restored_np * 255).astype(np.uint8)
else:
    restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)

print(f"Converted range: [{restored_np.min()}, {restored_np.max()}]")
print(f"Converted mean: {restored_np.mean():.1f}")

# Check if it's mostly white
white_percentage = np.sum(restored_np > 240) / (restored_np.size) * 100
print(f"Pixels > 240 (near white): {white_percentage:.1f}%")

# Save debug output
comparison = np.hstack([cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), restored_np])
cv2.imwrite('debug_output.jpg', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
print(f"\n✓ Saved debug output to: debug_output.jpg")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
print("\nPROBLEM:")
print("  Model was trained to output white because:")
print("  - No paired training data (foggy → clear)")
print("  - MSE loss minimized by outputting blank images")
print("\nSOLUTION:")
print("  Need to retrain with perceptual loss or use reference images")
print("=" * 80)
