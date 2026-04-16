"""
Streamlit Web Interface for Foggy Object Detection
Uses YOLOv8 for real-time object detection
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import os
import sys

st.set_page_config(page_title="Foggy Object Detection", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1E88E5;
    }
    .detection-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌫️ Foggy Object Detection System")
    st.markdown("**Advanced object detection for foggy weather conditions**")

with col2:
    st.markdown("**Status:**")
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"Running on {device_type}")

st.divider()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    confidence = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IOU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
    model_size = st.selectbox("Model Size", ["nano", "small", "medium"])
    
    st.divider()
    st.markdown("**About**")
    st.markdown("""
    This application uses:
    - **YOLOv8** for fast object detection
    - **Real-time processing** for immediate results
    - **Multiple model sizes** for different needs
    """)

# Load Model
@st.cache_resource
def load_yolo_model(size="nano"):
    """Load YOLOv8 model with caching"""
    try:
        # Suppress warnings
        import warnings
        warnings.filterwarnings('ignore')
        
        # Import here to avoid issues
        from ultralytics import YOLO
        
        model_map = {
            "nano": "yolov8n.pt",
            "small": "yolov8s.pt", 
            "medium": "yolov8m.pt"
        }
        
        model_name = model_map[size]
        st.write(f"📦 Loading YOLOv8 {size} model...")
        
        model = YOLO(model_name)
        st.write(f"✅ Model loaded successfully: {model_name}")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
try:
    model = load_yolo_model(model_size)
    if model is None:
        st.stop()
except Exception as e:
    st.error(f"Failed to initialize: {str(e)}")
    st.stop()

st.success("✅ Model ready for inference")
st.divider()

# Main Content
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Camera Feed", "📊 Batch Detection"])

# ========== TAB 1: Upload Image ==========
with tab1:
    st.header("Upload and Detect")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
    
    with col2:
        use_example = st.checkbox("Use Example Image", value=False)
    
    if use_example or uploaded_file is not None:
        # Get image
        if use_example:
            st.info("Using example image...")
            example_url = "https://images.pexels.com/photos/9993/nature-forest-trees.jpeg"
            image = Image.open(requests_get(example_url).raw if 'requests_get' in dir() else 'pexels-photo-19447954.jpeg')
        else:
            image = Image.open(uploaded_file)
        
        # Display original
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Run detection
        if st.button("🔍 Run Detection", key="detect_button", use_container_width=True):
            st.info("Processing image...")
            
            # Convert image to array
            image_np = np.array(image)
            if len(image_np.shape) == 2:  # Grayscale
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            
            # Run inference
            try:
                with st.spinner("Running detection..."):
                    results = model(image_np, conf=confidence, iou=iou_threshold)
                
                # Get annotated image
                result = results[0]
                annotated_frame = result.plot()
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Detection Results")
                    st.image(annotated_frame, use_column_width=True, channels="BGR")
                
                with col2:
                    st.subheader("Detected Objects")
                    
                    boxes = result.boxes
                    
                    if len(boxes) > 0:
                        st.success(f"✅ Found {len(boxes)} object(s)")
                        
                        detection_data = []
                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = result.names[cls_id]
                            
                            detection_data.append({
                                "ID": i + 1,
                                "Class": class_name,
                                "Confidence": f"{conf:.2%}",
                                "Coordinates": f"{box.xyxy[0].tolist()}"
                            })
                        
                        st.dataframe(detection_data, use_container_width=True)
                    else:
                        st.warning("No objects detected in this image")
                
            except Exception as e:
                st.error(f"Detection failed: {str(e)}")
                st.info("Try uploading a different image or adjusting the confidence threshold")

# ========== TAB 2: Information ==========
with tab2:
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Model Details
        - **Architecture**: YOLOv8
        - **Framework**: PyTorch
        - **Detection Classes**: 80 (COCO dataset)
        - **Real-time Processing**: Yes
        """)
    
    with col2:
        st.markdown(f"""
        ### Current Settings
        - **Model Size**: {model_size.upper()}
        - **Confidence**: {confidence}
        - **IOU Threshold**: {iou_threshold}
        - **Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}
        """)
    
    st.divider()
    st.markdown("""
    ### Supported Classes (COCO)
    person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
    fire hydrant, stop sign, parking meter, bench, cat, dog, horse, sheep, cow, elephant,
    bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis,
    snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
    tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
    sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
    potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard,
    microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
    teddy bear, hair drier, toothbrush
    """)

# ========== TAB 3: Batch Info ==========  
with tab3:
    st.header("📈 System Performance")
    
    st.info("""
    ### Performance Metrics
    - **Detection Speed**: ~30-150ms per image (varies by model size)
    - **Supported Resolutions**: 320x320 to 1280x1280
    - **Inference Framework**: NVIDIA CUDA / CPU
    """)
    
    st.markdown("### Tips for Best Results")
    st.markdown("""
    1. **Clear Images**: Better quality images produce better detections
    2. **Confidence Threshold**: Lower values detect more objects but may include false positives
    3. **Lighting**: Good lighting improves detection accuracy
    4. **Object Size**: Objects should be reasonably sized (not too small)
    5. **Model Selection**: 
       - Nano: Fast, lower accuracy
       - Small: Balanced
       - Medium: More accurate but slower
    """)

st.divider()

# Footer
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**MSFFA-YOLO System**")
with col2:
    st.markdown("**Foggy Weather Detection**")
with col3:
    st.markdown("**Real-time Processing**")
