import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import requests
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kinetic Vision Pro", page_icon="🧬", layout="wide")

# Persistent session state
if 'takeoff_f' not in st.session_state: st.session_state.takeoff_f = None
if 'landing_f' not in st.session_state: st.session_state.landing_f = None
if 'scrub_idx' not in st.session_state: st.session_state.scrub_idx = 0

st.sidebar.title("🚀 Control Hub")
analysis_mode = st.sidebar.radio("Analysis Mode", ["Auto (AI Agent)", "Manual (Frame Scrubber)"])
body_mass = st.sidebar.number_input("Athlete Mass (kg)", value=75.0)

st.title("🧬 Kinetic Vision: Biomechanics Engine")
video_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

if video_file:
    # Use a persistent path to ensure file availability
    tfile_path = os.path.join(tempfile.gettempdir(), "kinetic_video.mp4")
    with open(tfile_path, "wb") as f:
        f.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # CALCULATE FPS
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    if detected_fps < 60: detected_fps = 240.0 # Standard fallback
    
    st.sidebar.subheader("Physics Calibration")
    # This value is the key to fixing the 7cm discrepancy.
    # Set this to EXACTLY what your local version uses (e.g., 218.4)
    real_fps = st.sidebar.number_input("Confirmed Capture FPS", value=float(detected_fps), 
                                       help="If results differ from local, match this FPS value to your local version.")

    if analysis_mode == "Manual (Frame Scrubber)":
        st.subheader("🖱️ Manual Event Selection")
        m_col1, m_col2 = st.columns([2, 1])
        with m_col2:
            def update_frame(delta):
                st.session_state.scrub_idx = max(0, min(total_frames - 1, st.session_state.scrub_idx + delta))
            
            st.write("**Frame Navigation**")
            c1, c2 = st.columns(2)
            c1.button("⬅️ -1", on_click=update_frame, args=(-1,))
            c2.button("+1 ➡️", on_click=update_frame, args=(1,))
            st.slider("Scrubber", 0, total_frames - 1, key="scrub_idx")
            
            st.divider()
            b1, b2 = st.columns(2)
            if b1.button("📌 Set Takeoff"): st.session_state.takeoff_f = st.session_state.scrub_idx
            if b2.button("📌 Set Landing"): st.session_state.landing_f = st.session_state.scrub_idx
            
            if st.button("🔄 Reset"):
                st.session_state.takeoff_f = st.session_state.landing_f = None
                st.rerun()

        with m_col1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.scrub_idx)
            ret, frame = cap.read()
            if ret: st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        if st.session_state.takeoff_f and st.session_state.landing_f:
            f_frames = abs(st.session_state.landing_f - st.session_state.takeoff_f)
            f_time = f_frames / real_fps
            h_cm = (9.81 * (f_time**2) / 8) * 100
            st.success(f"### 📊 Result: {h_cm:.2f} cm")

    else:
        st.subheader("🤖 AI Automated Analysis")
        if st.button("🚀 Start AI Extraction"):
            model_path = 'pose_landmarker_lite.task'
            if not os.path.exists(model_path):
                r = requests.get("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
                with open(model_path, 'wb') as f: f.write(r.content)

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO)

            toe_y, valid_frames = [], []
            
            # TRACKING PASS: No frame storage to keep memory low
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                pbar = st.progress(0)
                for f_idx in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Process frame for AI
                    mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    res = landmarker.detect_for_video(mi, int((f_idx / 30.0) * 1000))
                    
                    if res.pose_landmarks:
                        l = res.pose_landmarks[0]
                        toe_y.append(1.0 - (l[31].y + l[32].y) / 2.0)
                        valid_frames.append(f_idx)
                    
                    if f_idx % 20 == 0: pbar.progress(f_idx / total_frames)
                pbar.empty()

            if len(toe_y) > 50:
                y_smooth = np.convolve(toe_y, np.ones(3)/3, mode='same')
                baseline = np.mean(y_smooth[:30])
                air = np.where(y_smooth > (baseline + 0.01))[0]
                
                if len(air) > 0:
                    jump = np.split(air, np.where(np.diff(air) > 5)[0] + 1)[-1]
                    t_off, l_nd = valid_frames[jump[0]], valid_frames[jump[-1]]
                    
                    # PHYSICS CALCULATION
                    f_frames = l_nd - t_off
                    f_time = f_frames / real_fps
                    h_cm = (9.81 * (f_time**2) / 8) * 100
                    
                    st.success(f"### 📐 AI Result: {h_cm:.2f} cm")
                    st.info(f"Using {f_frames} frames at {real_fps} FPS")

                    # DISPLAY RESULTS (Re-opening cap for just two frames is memory safe)
                    st.subheader("📸 AI Event Verification")
                    v1, v2 = st.columns(2)
                    
                    def get_frame(idx):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        r, img = cap.read()
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if r else None

                    v1.image(get_frame(t_off), caption=f"Takeoff (Frame {t_off})")
                    v2.image(get_frame(l_nd), caption=f"Landing (Frame {l_nd})")
                else: st.warning("Jump not detected.")

    cap.release()
    if os.path.exists(tfile_path): os.remove(tfile_path)
