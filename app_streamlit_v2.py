"""
Streamlit Web Interface for Foggy Object Detection
Uses YOLOv8 for real-time object detection
Voice-over feature powered by gTTS (Google Text-to-Speech)
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import os
import io
import base64

st.set_page_config(page_title="Foggy Object Detection", layout="wide", initial_sidebar_state="expanded")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { padding: 2rem; }
    .stTitle { color: #1E88E5; }
    .detection-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    /* Voice-over card */
    .voiceover-card {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border: 1px solid rgba(100, 120, 255, 0.35);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(50, 50, 150, 0.25);
        color: #fff;
    }
    .voiceover-card h3 {
        margin: 0 0 0.3rem 0;
        font-size: 1.1rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .voiceover-card p {
        font-size: 0.85rem;
        color: #c4b5fd;
        margin: 0 0 1rem 0;
    }
    .voiceover-script {
        background: rgba(255,255,255,0.06);
        border-left: 3px solid #7c3aed;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        line-height: 1.75;
        color: #e0d7ff;
        margin-bottom: 1rem;
        white-space: pre-line;
    }
    .audio-wrapper audio {
        width: 100%;
        border-radius: 8px;
        filter: drop-shadow(0 0 6px rgba(124,58,237,0.4));
    }

    @keyframes pulse-glow {
        0%   { box-shadow: 0 0 0 0 rgba(124,58,237,0.5); }
        70%  { box-shadow: 0 0 0 10px rgba(124,58,237,0); }
        100% { box-shadow: 0 0 0 0 rgba(124,58,237,0); }
    }
    .pulse-badge {
        display: inline-block;
        background: #7c3aed;
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        padding: 0.2rem 0.6rem;
        border-radius: 50px;
        animation: pulse-glow 2s infinite;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialisation ─────────────────────────────────────────────
# All detection results are stored here so they SURVIVE button re-clicks
for _key, _val in {
    "detection_data": None,       # list of dicts or [] when done
    "annotated_frame": None,      # numpy BGR array
    "image_displayed": None,      # PIL image shown to user
    "vo_mp3": None,               # bytes of last voice-over MP3
    "vo_script": None,            # text of last voice-over script
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_voiceover_script(detection_data: list) -> str:
    """Build a natural-language narration script from detection results."""
    if not detection_data:
        return (
            "No objects were detected in this image. "
            "Try lowering the confidence threshold or using a clearer image."
        )

    total = len(detection_data)
    lines = [f"Detection complete. {total} object{'s' if total != 1 else ''} found in the image."]

    class_confs: dict[str, list[float]] = {}
    for d in detection_data:
        class_confs.setdefault(d["Class"], []).append(d["_conf_raw"])

    for cls, confs in class_confs.items():
        count = len(confs)
        avg = sum(confs) / count
        pct = avg * 100
        if avg >= 0.75:
            verdict = "high confidence — very likely accurate"
        elif avg >= 0.50:
            verdict = "moderate confidence — probably accurate"
        else:
            verdict = "low confidence — may be a false positive"
        noun = f"{count} {cls}{'s' if count > 1 else ''}"
        lines.append(
            f"{noun} detected, average confidence {pct:.0f} percent. "
            f"This is {verdict}."
        )

    overall = sum(d["_conf_raw"] for d in detection_data) / total
    if overall >= 0.70:
        lines.append("Overall, the detection results appear to be reliable.")
    elif overall >= 0.45:
        lines.append("Overall confidence is moderate. Consider reviewing detections manually.")
    else:
        lines.append(
            "Overall confidence is low. "
            "Consider adjusting the threshold or using a higher quality image."
        )
    return " ".join(lines)


def generate_mp3(script: str) -> bytes | None:
    """Use gTTS to convert script → MP3 bytes."""
    try:
        from gtts import gTTS
        tts = gTTS(text=script, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error(f"Voice-over error: {e}")
        return None


def audio_html(mp3_bytes: bytes) -> str:
    """Embed MP3 as an auto-playing HTML audio element."""
    b64 = base64.b64encode(mp3_bytes).decode()
    return (
        '<div class="audio-wrapper">'
        '<audio controls autoplay>'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        'Your browser does not support the audio element.'
        '</audio>'
        '</div>'
    )


def render_voiceover_ui(detection_data: list):
    """
    Render the 🔊 voice-over button + result card.
    Called OUTSIDE the Run-Detection if-block so it persists across reruns.
    """
    st.markdown("---")
    st.markdown(
        '<span style="font-size:0.95rem;color:#7c3aed;font-weight:600;">'
        '🎙️ Voice-Over Narration'
        '<span class="pulse-badge">NEW</span>'
        '</span>',
        unsafe_allow_html=True,
    )

    # If we already have cached MP3, show player + re-generate option
    if st.session_state.vo_mp3 is not None:
        _show_voiceover_card(st.session_state.vo_script, st.session_state.vo_mp3)
        st.button(
            "🔄 Re-generate Voice-Over",
            key="vo_regen",
            on_click=_do_voiceover,
            args=(detection_data,),
            use_container_width=True,
        )
    else:
        st.button(
            "🔊 Generate & Play Voice-Over",
            key="vo_generate",
            on_click=_do_voiceover,
            args=(detection_data,),
            use_container_width=True,
        )


def _do_voiceover(detection_data: list):
    """Callback – runs BEFORE the next render, stores result in session_state."""
    script = build_voiceover_script(detection_data)
    mp3 = generate_mp3(script)
    st.session_state.vo_script = script
    st.session_state.vo_mp3 = mp3


def _show_voiceover_card(script: str, mp3_bytes: bytes):
    st.markdown(
        f"""
        <div class="voiceover-card">
            <h3>🎙️ Voice-Over Narration</h3>
            <p>AI-generated speech narrating your detection results.</p>
            <div class="voiceover-script">{script}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(audio_html(mp3_bytes), unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Download Voice-Over (MP3)",
        data=mp3_bytes,
        file_name="detection_voiceover.mp3",
        mime="audio/mp3",
        use_container_width=True,
        key="vo_download",
    )


# ─── Title & Header ───────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌫️ Foggy Object Detection System")
    st.markdown("**Advanced object detection for foggy weather conditions**")
with col2:
    st.markdown("**Status:**")
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"Running on {device_type}")

st.divider()

# ─── Sidebar Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    confidence   = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IOU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
    model_size   = st.selectbox("Model Size", ["nano", "small", "medium"])
    st.divider()
    st.markdown("**About**")
    st.markdown("""
    This application uses:
    - **YOLOv8** for fast object detection
    - **Real-time processing** for immediate results
    - **Multiple model sizes** for different needs
    - **gTTS Voice-over** to narrate detection results
    """)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo_model(size="nano"):
    import warnings; warnings.filterwarnings('ignore')
    from ultralytics import YOLO
    model_map = {"nano": "yolov8n.pt", "small": "yolov8s.pt", "medium": "yolov8m.pt"}
    model_name = model_map[size]
    st.write(f"📦 Loading YOLOv8 {size} model...")
    model = YOLO(model_name)
    st.write(f"✅ Model loaded: {model_name}")
    return model

try:
    model = load_yolo_model(model_size)
    if model is None:
        st.stop()
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

st.success("✅ Model ready for inference")
st.divider()

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Camera Feed", "📊 Batch Detection"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – Upload Image
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Upload and Detect")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="image_uploader",
        )
    with col2:
        use_example = st.checkbox("Use Example Image", value=False)

    # ── Resolve the input image ───────────────────────────────────────────────
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif use_example:
        example_path = "pexels-photo-19447954.jpeg"
        if os.path.exists(example_path):
            image = Image.open(example_path)
            st.info("Using example image…")
        else:
            st.error("Example image not found. Please upload an image instead.")

    if image is not None:
        # Show original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # ── Run Detection button ──────────────────────────────────────────────
        if st.button("🔍 Run Detection", key="detect_button", use_container_width=True):
            # Reset previous voice-over when a fresh detection is run
            st.session_state.vo_mp3    = None
            st.session_state.vo_script = None

            st.info("Processing image…")
            image_np = np.array(image)
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

            try:
                with st.spinner("Running detection…"):
                    results = model(image_np, conf=confidence, iou=iou_threshold)

                result          = results[0]
                annotated_frame = result.plot()
                boxes           = result.boxes

                # Build detection_data
                detection_data = []
                for i, box in enumerate(boxes):
                    cls_id     = int(box.cls[0])
                    conf_val   = float(box.conf[0])
                    class_name = result.names[cls_id]
                    detection_data.append({
                        "ID":          i + 1,
                        "Class":       class_name,
                        "Confidence":  f"{conf_val:.2%}",
                        "Coordinates": str(box.xyxy[0].tolist()),
                        "_conf_raw":   conf_val,      # hidden key for TTS
                    })

                # Persist in session state so voice-over button survives rerun
                st.session_state.detection_data   = detection_data
                st.session_state.annotated_frame  = annotated_frame

            except Exception as e:
                st.error(f"Detection failed: {e}")
                st.info("Try a different image or adjust the confidence threshold.")

        # ── Always render results if we have them (survives reruns) ──────────
        if st.session_state.detection_data is not None:
            detection_data  = st.session_state.detection_data
            annotated_frame = st.session_state.annotated_frame

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.subheader("Detection Results")
                st.image(annotated_frame, use_column_width=True, channels="BGR")

            with res_col2:
                st.subheader("Detected Objects")

                if len(detection_data) > 0:
                    st.success(f"✅ Found {len(detection_data)} object(s)")
                    display_data = [
                        {k: v for k, v in d.items() if not k.startswith("_")}
                        for d in detection_data
                    ]
                    st.dataframe(display_data, use_container_width=True)
                else:
                    st.warning("No objects detected in this image.")

                # ── Voice-Over UI (always visible once detection has run) ──────
                render_voiceover_ui(detection_data)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model Information
# ═════════════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – System Performance
# ═════════════════════════════════════════════════════════════════════════════
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
    4. **Object Size**: Objects should be reasonably sized in frame
    5. **Model Selection**:
       - Nano: Fast, lower accuracy
       - Small: Balanced
       - Medium: More accurate but slower
    """)

st.divider()

# ─── Footer ───────────────────────────────────────────────────────────────────
fc1, fc2, fc3 = st.columns(3)
with fc1: st.markdown("**MSFFA-YOLO System**")
with fc2: st.markdown("**Foggy Weather Detection**")
with fc3: st.markdown("**Real-time Processing**")
