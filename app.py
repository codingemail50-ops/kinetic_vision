import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import requests
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kinetic Vision AI",
    page_icon="🧬",
    layout="wide"
)

# --- HEADER & PROTOCOL ---
st.title("🧬 Kinetic Vision: Biomechanics Engine")
st.markdown("**Automated Vertical Displacement & Reactive Strength Analysis**")

with st.expander("📋 Experimental Protocol (Read for Accuracy)", expanded=False):
    st.markdown("""
    **Device & Environment Setup:**
    
    1.  **High-Speed Capture:** Recording **MUST** be in **Slow Motion** (120 or 240 FPS).
    2.  **Camera Position (CRITICAL):**
        * **SIDE PROFILE (Sagittal Plane):** Camera must be perpendicular to the athlete.
        * **Static Mount:** Prop phone against a wall. Do not hold by hand.
    3.  **Subject Preparation:**
        * **Contrast:** Wear attire that contrasts with the background.
        * **Lighting:** High ambient light required.
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
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    video_file = st.file_uploader("1. Ingest Video Sample (Side Profile)", type=['mp4', 'mov', 'avi'])

with col2:
    real_fps = st.selectbox(
        "2. Temporal Resolution (FPS)",
        options=[30, 60, 120, 240],
        index=3, 
        help="Select 240 for standard iOS/Android Slow-Mo."
    )

with col3:
    body_mass = st.number_input(
        "3. Athlete Mass (kg)",
        min_value=30, 
        max_value=150, 
        value=75,
        help="Required for Peak Power (Watts) calculation."
    )

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    file_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                timestamp_ms = int((frame_idx / file_fps) * 1000)
                
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if detection_result.pose_landmarks:
                    landmarks = detection_result.pose_landmarks[0]
                    y_pos = 1.0 - landmarks[27].y # Invert Y so up is positive
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
    # We map data to indices to keep track of frames correctly
    # Replace None with NaN for plotting breaks
    y_plot = np.array([y if y is not None else np.nan for y in ankle_y_values])
    
    # Create a clean version for math (no NaNs)
    valid_indices = [i for i, y in enumerate(ankle_y_values) if y is not None]
    valid_values = [ankle_y_values[i] for i in valid_indices]
    
    if len(valid_values) > 30:
        y_clean = np.array(valid_values)
        
        # Smoothing
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        y_smooth = np.convolve(y_clean, kernel, mode='same')
        
        # Baseline & Threshold
        baseline = np.mean(y_smooth[:30])
        threshold = baseline + 0.02 
        
        in_air_indices = np.where(y_smooth > threshold)[0]
        
        if len(in_air_indices) > 0:
            # Event Detection
            splits = np.split(in_air_indices, np.where(np.diff(in_air_indices) > 5)[0] + 1)
            jump_event = max(splits, key=len)
            
            # These are indices relative to y_clean/y_smooth
            takeoff_idx_local = jump_event[0]
            landing_idx_local = jump_event[-1]
            
            # Map back to GLOBAL frame numbers (for images)
            takeoff_frame = valid_indices[takeoff_idx_local]
            landing_frame = valid_indices[landing_idx_local]
            
            flight_frames = landing_frame - takeoff_frame
            flight_time = flight_frames / real_fps
            
            height_m = (9.81 * (flight_time**2)) / 8
            height_cm = height_m * 100
            
            # RSI Logic
            start_search_idx = max(0, takeoff_idx_local - int(1.5 * real_fps))
            pre_takeoff_slice = y_smooth[start_search_idx:takeoff_idx_local]
            
            if len(pre_takeoff_slice) > 0:
                dip_idx_relative = np.argmin(pre_takeoff_slice)
                dip_idx_local = start_search_idx + dip_idx_relative
                dip_frame_global = valid_indices[dip_idx_local]
                
                time_to_takeoff = (takeoff_frame - dip_frame_global) / real_fps
                if time_to_takeoff < 0.1: time_to_takeoff = 0.4
                rsi_mod = height_m / time_to_takeoff
            else:
                rsi_mod = 0.0
                dip_idx_local = takeoff_idx_local # Fallback
                dip_frame_global = takeoff_frame

            peak_power = (60.7 * height_cm) + (45.3 * body_mass) - 2055

            # --- RESULTS ---
            st.divider()
            if 10 < height_cm < 120:
                st.success(f"### 📐 Vertical Displacement: {height_cm:.2f} cm")
                m1, m2, m3 = st.columns(3)
                m1.metric("Flight Time", f"{flight_time:.3f} s")
                m2.metric("RSI-mod", f"{rsi_mod:.2f}")
                m3.metric("Peak Power", f"{int(peak_power)} W")
            else:
                st.warning(f"Calculated: {height_cm:.2f} cm")
                st.info("Check FPS selection.")
            
            # --- GRAPHING ENGINE (MATPLOTLIB) ---
            st.divider()
            st.subheader("📊 Biomechanical Trajectory")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            # Plot the smoothed curve
            # We use valid_indices for X-axis so it matches frame numbers
            ax.plot(valid_indices, y_smooth, label='Ankle Vertical Velocity', color='#2980b9', linewidth=2)
            
            # Annotate Key Points
            apex_idx_local = takeoff_idx_local + (len(jump_event) // 2)
            
            # Get (X, Y) coordinates for the dots
            x_dip = valid_indices[dip_idx_local]
            y_dip = y_smooth[dip_idx_local]
            
            x_takeoff = valid_indices[takeoff_idx_local]
            y_takeoff = y_smooth[takeoff_idx_local]
            
            x_apex = valid_indices[apex_idx_local]
            y_apex = y_smooth[apex_idx_local]
            
            # Plot Dots
            ax.scatter([x_dip], [y_dip], color='red', s=100, zorder=5, label='Eccentric (Dip)')
            ax.scatter([x_takeoff], [y_takeoff], color='gold', s=100, zorder=5, label='Concentric (Takeoff)')
            ax.scatter([x_apex], [y_apex], color='green', s=100, zorder=5, label='Apex')
            
            # Styling
            ax.set_ylabel("Vertical Position (Normalized)")
            ax.set_xlabel("Frame Number")
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Render in Streamlit
            st.pyplot(fig)

            # --- ANNOTATION ENGINE (IMAGES) ---
            st.divider()
            st.subheader("📸 Biomechanical Keyframes")
            
            apex_frame_global = valid_indices[apex_idx_local]
            
            targets = {
                dip_frame_global: ("ECCENTRIC (DIP)", (0, 0, 255)),       
                takeoff_frame:    ("CONCENTRIC (TAKEOFF)", (0, 255, 255)), 
                apex_frame_global: (f"APEX ({height_cm:.1f}cm)", (0, 255, 0)) 
            }
            
            cap.release() 
            cap = cv2.VideoCapture(tfile.name)
            captured_images = {}
            curr_frame = 0
            max_target = max(targets.keys())
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if curr_frame in targets:
                    label, color = targets[curr_frame]
                    cv2.putText(frame, label, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,0), 15, cv2.LINE_AA)
                    cv2.putText(frame, label, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4, cv2.LINE_AA)
                    captured_images[curr_frame] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                curr_frame += 1
                if curr_frame > max_target: break
            
            cap.release()
            
            c1, c2, c3 = st.columns(3)
            if dip_frame_global in captured_images:
                c1.image(captured_images[dip_frame_global], use_container_width=True, caption="Phase 1: Loading")
            if takeoff_frame in captured_images:
                c2.image(captured_images[takeoff_frame], use_container_width=True, caption="Phase 2: Propulsion")
            if apex_frame_global in captured_images:
                c3.image(captured_images[apex_frame_global], use_container_width=True, caption="Phase 3: Max Height")

        else:
            st.warning("No significant vertical displacement detected.")