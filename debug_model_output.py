"""
Debug script to check model output format and values
"""
import torch
import numpy as np
import cv2
from models.msffa_yolo import MSFFA_YOLO

# Load model
device = "cpu"
model = MSFFA_YOLO(num_classes=8).to(device)
model.eval()

# Create dummy input (1, 3, 512, 512)
dummy_input = torch.randn(1, 3, 512, 512).to(device)

print("=" * 60)
print("DEBUGGING MODEL OUTPUT")
print("=" * 60)
print(f"Input shape: {dummy_input.shape}")
print(f"Input dtype: {dummy_input.dtype}")
print(f"Input range: [{dummy_input.min():.4f}, {dummy_input.max():.4f}]")

with torch.no_grad():
    yolo_preds, restored_img = model(dummy_input)

print(f"\nRestored image shape: {restored_img.shape}")
print(f"Restored image dtype: {restored_img.dtype}")
print(f"Restored image range: [{restored_img.min():.4f}, {restored_img.max():.4f}]")

# Try to convert like in app
restored_np = restored_img[0].permute(1, 2, 0).cpu().numpy()
print(f"\nAfter permute and numpy:")
print(f"  Shape: {restored_np.shape}")
print(f"  Dtype: {restored_np.dtype}")
print(f"  Range: [{restored_np.min():.4f}, {restored_np.max():.4f}]")

# Apply conversion
if restored_np.min() < 0:  # Tanh output is in [-1, 1]
    print("\nDetected Tanh output [-1, 1], converting...")
    restored_np = (restored_np + 1) / 2.0  # Convert to [0, 1]
    print(f"  After Tanh->normalized: range [{restored_np.min():.4f}, {restored_np.max():.4f}]")
    restored_np = (restored_np * 255).astype(np.uint8)
    print(f"  After scaling to [0-255]: range [{restored_np.min()}, {restored_np.max()}]")
else:
    print(f"\nNOT Tanh output! Range is {restored_np.min():.4f} to {restored_np.max():.4f}")
    if restored_np.max() <= 1.0:
        restored_np = (restored_np * 255).astype(np.uint8)
    else:
        restored_np = np.clip(restored_np, 0, 255).astype(np.uint8)

print(f"\nFinal numpy array:")
print(f"  Shape: {restored_np.shape}")
print(f"  Dtype: {restored_np.dtype}")
print(f"  Range: [{restored_np.min()}, {restored_np.max()}]")
print(f"  Sample values:\n{restored_np[0:2, 0:3, :]}")

# Save visualization
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 1, figsize=(8, 6))
axes.imshow(restored_np)
axes.set_title("Model Output Visualization")
axes.axis('off')
plt.tight_layout()
plt.savefig('debug_output.png', dpi=100, bbox_inches='tight')
print(f"\n✓ Visualization saved to debug_output.png")

print("\n" + "=" * 60)
