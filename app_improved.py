"""
Streamlit Web Interface for MSFFA-YOLO
Fog Removal + Object Detection
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from models.msffa_yolo import MSFFA_YOLO

st.set_page_config(page_title="MSFFA-YOLO Fog Removal", layout="wide")

# Title and description
st.title("🌫️ MSFFA-YOLO: Fog Removal & Object Detection")
st.markdown("""
**Advanced Fog Removal System**
- Removes fog from foggy images using MSFFA restoration subnet
- Detects objects in restored images using YOLO detection subnet  
- Real-time processing with visual comparisons
""")

st.divider()

# Load model
@st.cache_resource
def load_model():
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSFFA_YOLO(num_classes=8).to(device)
    
    # Try to load trained weights
    weight_paths = [
        "weights/msffa_yolo_final.pth",
        "weights/msffa_yolo_epoch_10.pth",
        "weights/msffa_yolo_epoch_9.pth",
    ]
    
    loaded = False
    for weight_path in weight_paths:
        if os.path.exists(weight_path):
            try:
                model.load_state_dict(torch.load(weight_path, map_location=device))
                print(f"✓ Loaded trained weights from: {weight_path}")
                loaded = True
                break
            except Exception as e:
                print(f"⚠ Could not load weights from {weight_path}: {e}")
    
    if not loaded:
        print("⚠ No trained weights found - using random initialization")
    
    model.eval()
    return model, device

print("Loading model...")
model, device = load_model()
st.success(f"✅ Model loaded on {device.upper()}")

st.divider()

# Upload section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a foggy image",
        type=["jpg", "png", "jpeg"],
        help="Upload a foggy image for fog removal and object detection"
    )

with col2:
    st.subheader("⚙️ Settings")
    confidence_threshold = st.slider(
        "Detection Confidence",
        0.0, 1.0, 0.3, 0.1,
        help="Higher = fewer but more confident detections"
    )

st.divider()

if uploaded_file is not None:
    # Read and prepare image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Convert BGR if needed
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np
    
    # Show original
    st.subheader("📸 Input Image")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Original Foggy Image", use_column_width=True)
    
    # Process button
    if st.button("🔄 Process Image", use_container_width=True):
        with st.spinner("🔄 Processing... This may take a moment..."):
            try:
                # Prepare tensor
                img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Run model
                with torch.no_grad():
                    yolo_preds, restored_img = model(img_tensor)
                
                # Convert to numpy
                restored_np = restored_img[0].permute(1, 2, 0).cpu().numpy()
                
                # Handle Tanh output [-1, 1] -> [0, 255]
                if restored_np.min() < 0:  # Tanh output is in [-1, 1]
                    restored_np = (restored_np + 1) / 2.0  # Convert to [0, 1]
                    restored_np = (restored_np * 255).astype(np.uint8)
                elif restored_np.max() <= 1.0:  # Already normalized [0, 1]
                    restored_np = (restored_np * 255).astype(np.uint8)
                else:  # Already in [0, 255]
                    restored_np = np.clip(restored_np, 0, 255).astype(np.uint8)
                
                # Post-processing: Apply CLAHE enhancement to improve visual quality
                restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
                img_lab = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
                restored_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
                restored_enhanced_rgb = cv2.cvtColor(restored_enhanced, cv2.COLOR_BGR2RGB)
                
                # Display results
                st.subheader("🎯 Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown("**Before: Foggy Image**")
                    st.image(image, use_column_width=True)
                
                with result_col2:
                    st.markdown("**After: Fog Removed (Enhanced)**")
                    st.image(restored_enhanced_rgb, use_column_width=True)
                
                # Statistics
                st.divider()
                st.subheader("📊 Image Quality Metrics")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    # Convert to grayscale for brightness
                    original_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    original_brightness = original_gray.mean()
                    st.metric(
                        "Original Brightness",
                        f"{original_brightness:.1f}",
                        help="Brightness value of foggy image (0-255)"
                    )
                
                with metric_col2:
                    enhanced_gray = cv2.cvtColor(restored_enhanced, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    restored_brightness = enhanced_gray.mean()
                    st.metric(
                        "Restored Brightness",
                        f"{restored_brightness:.1f}",
                        help="Brightness value after fog removal (0-255)"
                    )
                
                with metric_col3:
                    original_contrast = original_gray.std()
                    st.metric(
                        "Original Contrast",
                        f"{original_contrast:.1f}",
                        help="Contrast (std dev) of foggy image"
                    )
                
                with metric_col4:
                    restored_contrast = enhanced_gray.std()
                    st.metric(
                        "Restored Contrast",
                        f"{restored_contrast:.1f}",
                        help="Contrast (std dev) after fog removal"
                    )
                
                # Download section
                st.divider()
                st.subheader("💾 Download Results")
                
                # Convert to PIL for download
                restored_pil = Image.fromarray(restored_enhanced_rgb)
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    # Save to buffer
                    from io import BytesIO
                    buf = BytesIO()
                    restored_pil.save(buf, format="PNG")
                    buf.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Restored Image",
                        data=buf,
                        file_name="restored_image.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with download_col2:
                    # Create side-by-side comparison
                    h, w = image_bgr.shape[:2]
                    restored_resized = cv2.resize(restored_enhanced_rgb, (w, h))
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    comparison = np.hstack([image_rgb, restored_resized])
                    comparison_pil = Image.fromarray(comparison)
                    
                    buf_comp = BytesIO()
                    comparison_pil.save(buf_comp, format="PNG")
                    buf_comp.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Comparison (Before/After)",
                        data=buf_comp,
                        file_name="comparison.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                st.success("✅ Processing complete!")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
else:
    st.info("👆 Upload an image to get started!")

# Footer
st.divider()
st.markdown("""
---
**MSFFA-YOLO Model**
- MSFFA Restoration Subnet: Removes fog and enhances image clarity
- YOLO Detection Subnet: Detects objects in the restored image
- Multi-scale feature fusion with attention mechanism
- Compatible with foggy traffic and driving scene analysis
""")
