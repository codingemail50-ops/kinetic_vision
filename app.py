import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kinetic Vision AI",
    page_icon="🧬",
    layout="centered"
)

# --- HEADER & PROTOCOL ---
st.title("🧬 Kinetic Vision: Biomechanics Engine")
st.markdown("**Automated Vertical Displacement Analysis via Computer Vision**")

with st.expander("📋 Experimental Protocol (Read for Accuracy)", expanded=True):
    st.markdown("""
    **Device & Environment Setup:**
    
    1.  **High-Speed Capture:** * Recording **MUST** be in **Slow Motion** (120 or 240 FPS).
        * Standard 30 FPS video lacks the temporal resolution for scientific measurement.
    
    2.  **Camera Position:**
        * **Static Mount:** Camera must be stationary (prop against a wall).
        * **Field of View:** Ensure full lower-body visibility throughout the movement range.
    
    3.  **Subject Preparation:**
        * **Contrast:** Subject should wear attire that contrasts with the background.
        * **Lighting:** High ambient light is required to minimize motion blur artifacts.
    """)

# --- 1. MODEL INITIALIZATION ---
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    with st.spinner("Initializing Neural Network..."):
        try:
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            r = requests.get(url, allow_redirects=True)
            with open(model_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            st.error(f"Model Load Failure: {e}")

# --- 2. DATA INGESTION ---
st.divider()
col1, col2 = st.columns([2, 1])

with col1:
    video_file = st.file_uploader("1. Ingest Video Sample", type=['mp4', 'mov', 'avi'])

with col2:
    # Manual FPS Override
    real_fps = st.selectbox(
        "2. Temporal Resolution (FPS)",
        options=[30, 60, 120, 240],
        index=3, 
        help="Select 240 for standard iOS/Android Slow-Mo. 30 for standard video."
    )

if video_file:
    # Process Temp File
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    file_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / file_fps
    
    # Metadata Diagnostic
    st.caption(f"**Input Diagnostic:** Source {int(file_fps)} FPS | Duration: {duration:.2f}s | Processing Target: **{real_fps} Hz**")

    # --- 3. COMPUTER VISION PIPELINE ---
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)

    ankle_y_values = []
    
    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB Conversion
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                # Sync Timestamp
                timestamp_ms = int((frame_idx / file_fps) * 1000)
                
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if detection_result.pose_landmarks:
                    landmarks = detection_result.pose_landmarks[0]
                    # Track Left Lateral Malleolus (Ankle)
                    y_pos = 1.0 - landmarks[27].y 
                    ankle_y_values.append(y_pos)
                else:
                    ankle_y_values.append(None)
                
                frame_idx += 1
                if frame_idx % 10 == 0:
                    progress_bar.progress(min(frame_idx / frame_count, 1.0))
                    status_text.text(f"Extracting Kinematics... Frame {frame_idx}/{frame_count}")
            
            progress_bar.empty()
            status_text.empty()
            
    except Exception as e:
        st.error(f"Processing Error: {e}")

    cap.release()

    # --- 4. PHYSICS ENGINE ---
    y_data = np.array(ankle_y_values)
    valid_data = [y for y in y_data if y is not None]
    
    if len(valid_data) > 30:
        y_clean = np.array(valid_data)
        
        # Calibration: Establish baseline from initial static frames
        baseline = np.mean(y_clean[:30])
        threshold = baseline + 0.02 
        
        in_air_indices = np.where(y_clean > threshold)[0]
        
        if len(in_air_indices) > 0:
            # Event Detection
            splits = np.split(in_air_indices, np.where(np.diff(in_air_indices) > 5)[0] + 1)
            jump_event = max(splits, key=len)
            
            flight_frames = jump_event[-1] - jump_event[0]
            
            # Physics Calculation (Newtonian)
            flight_time = flight_frames / real_fps
            
            height_m = (9.81 * (flight_time**2)) / 8
            height_cm = height_m * 100
            
            # --- RESULTS ---
            st.divider()
            
            # Range Validation
            if 10 < height_cm < 120:
                st.success(f"### 📐 Calculated Displacement: {height_cm:.2f} cm")
                st.metric("Flight Duration (t)", f"{flight_time:.3f} s")
            else:
                st.warning(f"Calculated: {height_cm:.2f} cm")
                st.info("Value outside physiological norms. Verify FPS selection.")
            
            # Data Visualization
            st.line_chart(y_clean)
            st.caption("Figure 1: Vertical Displacement of Left Ankle (Pixels)")

        else:
            st.warning("No significant vertical displacement detected.")
    else:
        st.error("Tracking Failed. Subject visibility compromised.")