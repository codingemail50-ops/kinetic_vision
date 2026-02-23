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

# --- SESSION STATE INITIALIZATION ---
if 'takeoff_f' not in st.session_state: st.session_state.takeoff_f = None
if 'landing_f' not in st.session_state: st.session_state.landing_f = None
if 'scrub_idx' not in st.session_state: st.session_state.scrub_idx = 0

st.sidebar.title("🚀 Control Hub")
analysis_mode = st.sidebar.radio("Analysis Mode", ["Auto (AI Agent)", "Manual (Frame Scrubber)"])
body_mass = st.sidebar.number_input("Athlete Mass (kg)", value=75.0, min_value=30.0)

st.title("🧬 Kinetic Vision: Biomechanics Engine")
video_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    real_fps = st.sidebar.number_input("Confirmed Capture FPS", value=240.0)

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

    else:
        st.subheader("🤖 AI Automated Analysis")
        if st.button("🚀 Start AI Extraction"):
            model_path = 'pose_landmarker_lite.task'
            if not os.path.exists(model_path):
                r = requests.get("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
                with open(model_path, 'wb') as f: f.write(r.content)

            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO)

            toe_y, valid_frames = [], []
            temp_frames = {} # Temporary storage for verification images
            
            with PoseLandmarker.create_from_options(options) as landmarker:
                pbar = st.progress(0)
                for f_idx in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    ts = int((f_idx / 30.0) * 1000)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    res = landmarker.detect_for_video(mp_image, ts)
                    
                    if res.pose_landmarks:
                        l = res.pose_landmarks[0]
                        toe_y.append(1.0 - (l[31].y + l[32].y) / 2.0)
                        valid_frames.append(f_idx)
                        
                        # Optimization: Store the drawn frame immediately during the main loop
                        # This prevents needing to "re-track" the same frame later
                        h, w, _ = frame.shape
                        for t in [l[31], l[32]]:
                            cv2.circle(frame, (int(t.x*w), int(t.y*h)), 10, (0, 255, 0), -1)
                        temp_frames[f_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if f_idx % 20 == 0: pbar.progress(f_idx / total_frames)
                
                if len(toe_y) > 50:
                    y_smooth = np.convolve(toe_y, np.ones(3)/3, mode='same')
                    baseline = np.mean(y_smooth[:30])
                    air = np.where(y_smooth > (baseline + 0.01))[0]
                    
                    if len(air) > 0:
                        jump = np.split(air, np.where(np.diff(air) > 5)[0] + 1)[-1]
                        t_off, l_nd = valid_frames[jump[0]], valid_frames[jump[-1]]
                        
                        st.subheader("📸 AI Event Verification")
                        v1, v2 = st.columns(2)
                        
                        # Fetch the pre-drawn images from our dictionary
                        if t_off in temp_frames:
                            v1.image(temp_frames[t_off], caption=f"Takeoff: Frame {t_off}")
                        if l_nd in temp_frames:
                            v2.image(temp_frames[l_nd], caption=f"Landing: Frame {l_nd}")
                        
                        f_time = (l_nd - t_off) / real_fps
                        h_cm = (9.81 * (f_time**2) / 8) * 100
                        st.success(f"### 📐 AI Result: {h_cm:.2f} cm")
                    else:
                        st.warning("No jump detected.")
    cap.release()
