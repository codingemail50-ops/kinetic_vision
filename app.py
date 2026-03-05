import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import requests
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Kinetic Vision Pro", page_icon="🧬", layout="wide")

# Unique session state initialization
if 'takeoff_f' not in st.session_state: st.session_state.takeoff_f = None
if 'landing_f' not in st.session_state: st.session_state.landing_f = None
if 'scrub_idx' not in st.session_state: st.session_state.scrub_idx = 0

st.sidebar.title("🚀 Control Hub")
analysis_mode = st.sidebar.radio("Mode", ["Auto (AI Agent)", "Manual (Scrubber)"])
body_mass = st.sidebar.number_input("Athlete Mass (kg)", value=75.0)

st.title("🧬 Kinetic Vision: Biomechanics Engine")
video_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

if video_file:
    # Unique temporary file per session to avoid multi-user collisions
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    if detected_fps < 10: detected_fps = 240.0
    
    real_fps = st.sidebar.number_input("Confirmed Capture FPS", value=float(detected_fps))

    if analysis_mode == "Manual (Scrubber)":
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
        if st.button("🚀 Start AI Extraction"):
            model_path = 'pose_landmarker_lite.task'
            if not os.path.exists(model_path):
                r = requests.get("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
                with open(model_path, 'wb') as f: f.write(r.content)

            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO)

            toe_y, valid_frames = [], []
            
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                pbar = st.progress(0)
                for f_idx in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    res = landmarker.detect_for_video(mp_image, int((f_idx / 30.0) * 1000))
                    
                    if res.pose_landmarks:
                        l = res.pose_landmarks[0]
                        # Tracking Foot Index/Toe Tip (Landmarks 31 and 32)
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
                    
                    f_time = (l_nd - t_off) / real_fps
                    h_cm = (9.81 * (f_time**2) / 8) * 100
                    st.success(f"### 📐 AI Result: {h_cm:.2f} cm")
                    
                    v1, v2 = st.columns(2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, t_off)
                    ret_t, img_t = cap.read()
                    if ret_t: v1.image(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB), caption="Takeoff")
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, l_nd)
                    ret_l, img_l = cap.read()
                    if ret_l: v2.image(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB), caption="Landing")
                else:
                    st.warning("No jump detected. Ensure athlete's feet are visible.")

    cap.release()
    if os.path.exists(video_path):
        try: os.remove(video_path)
        except: pass
