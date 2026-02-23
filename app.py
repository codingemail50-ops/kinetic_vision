import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import requests
import matplotlib.pyplot as plt
import subprocess
import json
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kinetic Vision Pro", page_icon="🧬", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'takeoff_f' not in st.session_state: st.session_state.takeoff_f = None
if 'landing_f' not in st.session_state: st.session_state.landing_f = None
if 'scrub_idx' not in st.session_state: st.session_state.scrub_idx = 0

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.title("🚀 Control Hub")
analysis_mode = st.sidebar.radio("Analysis Mode", ["Auto (AI Agent)", "Manual (Frame Scrubber)"])
body_mass = st.sidebar.number_input("Athlete Mass (kg)", value=75.0, min_value=30.0)

# --- HELPER: FPS DETECTOR ---
def get_true_capture_fps(file_path, filename_str=""):
    cap = cv2.VideoCapture(file_path)
    header_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if header_fps > 60: return header_fps, "Header"
    match = re.search(r'(\d{3})\s?fps', filename_str, re.IGNORECASE)
    if match: return float(match.group(1)), "Filename"
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})
        if 'com.apple.quicktime.capture.fps' in tags:
            return float(tags['com.apple.quicktime.capture.fps']), "Apple Metadata"
    except: pass
    return 240.0, "Default (Fallback)"

# --- MAIN UI ---
st.title("🧬 Kinetic Vision: Biomechanics Engine")
video_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    detected_fps, fps_source = get_true_capture_fps(tfile.name, video_file.name)
    real_fps = st.sidebar.number_input("Confirmed Capture FPS", value=detected_fps)
    st.sidebar.caption(f"Source: {fps_source}")

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    file_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # --- OPTION A: MANUAL MODE ---
    if analysis_mode == "Manual (Frame Scrubber)":
        st.subheader("🖱️ Manual Event Selection")
        m_col1, m_col2 = st.columns([2, 1])
        
        with m_col2:
            st.info("Pinpoint the frame where the toes break contact.")
            def update_frame(delta):
                new_val = st.session_state.scrub_idx + delta
                st.session_state.scrub_idx = max(0, min(total_frames - 1, new_val))

            st.write("**Frame Navigation**")
            nav_c1, nav_c2 = st.columns(2)
            nav_c1.button("⬅️ -1 Frame", on_click=update_frame, args=(-1,))
            nav_c2.button("+1 Frame ➡️", on_click=update_frame, args=(1,))
            
            st.slider("Coarse Scrubber", 0, total_frames - 1, key="scrub_idx")
            current_f = st.session_state.scrub_idx
            st.write(f"**Current Frame:** `{current_f}`")
            st.divider()
            
            b1, b2 = st.columns(2)
            if b1.button("📌 Set Takeoff"): st.session_state.takeoff_f = current_f
            if b2.button("📌 Set Landing"): st.session_state.landing_f = current_f
            
            st.write(f"**Takeoff:** `{st.session_state.takeoff_f if st.session_state.takeoff_f is not None else '---'}`")
            st.write(f"**Landing:** `{st.session_state.landing_f if st.session_state.landing_f is not None else '---'}`")

            if st.button("🔄 Reset"):
                st.session_state.takeoff_f = st.session_state.landing_f = None
                if "scrub_idx" in st.session_state: del st.session_state["scrub_idx"]
                st.rerun()

        with m_col1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.get("scrub_idx", 0))
            ret, frame = cap.read()
            if ret:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.session_state.takeoff_f is not None and st.session_state.landing_f is not None:
            if st.session_state.landing_f > st.session_state.takeoff_f:
                f_frames = st.session_state.landing_f - st.session_state.takeoff_f
                f_time = f_frames / real_fps
                h_cm = (9.81 * (f_time**2) / 8) * 100
                st.divider()
                st.success(f"### 📊 Manual Results: {h_cm:.2f} cm")
                st.metric("Flight Time", f"{f_time:.3f} s")

    # --- OPTION B: AUTO MODE ---
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
            preview_placeholder = st.empty() 

            with PoseLandmarker.create_from_options(options) as landmarker:
                pbar = st.progress(0)
                for f_idx in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    h, w, _ = frame.shape
                    timestamp_ms = int((f_idx / 30.0) * 1000) 
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    res = landmarker.detect_for_video(mp_image, timestamp_ms)
                    
                    if res.pose_landmarks:
                        landmarks = res.pose_landmarks[0]
                        avg_toe_y = (landmarks[31].y + landmarks[32].y) / 2.0
                        toe_y.append(1.0 - avg_toe_y)
                        valid_frames.append(f_idx)
                    
                    if f_idx % 20 == 0:
                        pbar.progress(f_idx / total_frames)
                
                pbar.empty()
                preview_placeholder.empty()

            if len(toe_y) > 50:
                y_smooth = np.convolve(toe_y, np.ones(3)/3, mode='same')
                baseline = np.mean(y_smooth[:30])
                air = np.where(y_smooth > (baseline + 0.01))[0]
                
                if len(air) > 0:
                    jump = np.split(air, np.where(np.diff(air) > 5)[0] + 1)[-1]
                    t_off_f, l_nd_f = valid_frames[jump[0]], valid_frames[jump[-1]]
                    f_frames = l_nd_f - t_off_f
                    f_time = f_frames / real_fps
                    h_cm = (9.81 * (f_time**2) / 8) * 100
                    
                    st.success(f"### 📐 Result: {h_cm:.2f} cm")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Takeoff Frame", t_off_f)
                    c2.metric("Landing Frame", l_nd_f)
                    c3.metric("Total Air Frames", f_frames)
                    
                    # --- NEW: BIOMECHANICAL VALIDATION IMAGES ---
                    st.divider()
                    st.subheader("📸 AI Event Verification")
                    iv_col1, iv_col2 = st.columns(2)
                    
                    # Re-open capture to grab the specific frames
                    cap_verify = cv2.VideoCapture(tfile.name)
                    
                    # Helper to draw circles on a specific frame
                    def get_annotated_frame(frame_idx):
                        cap_verify.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, img = cap_verify.read()
                        if not ret: return None
                        
                        # Briefly run AI on just this frame to get dot positions
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        res = landmarker.detect_for_video(mp_img, int((frame_idx/30.0)*1000))
                        if res.pose_landmarks:
                            h, w, _ = img.shape
                            l_toe, r_toe = res.pose_landmarks[0][31], res.pose_landmarks[0][32]
                            for t in [l_toe, r_toe]:
                                cv2.circle(img, (int(t.x*w), int(t.y*h)), 10, (0, 255, 0), -1)
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    takeoff_img = get_annotated_frame(t_off_f)
                    landing_img = get_annotated_frame(l_nd_f)
                    
                    if takeoff_img is not None:
                        iv_col1.image(takeoff_img, caption=f"AI Detected Takeoff (Frame {t_off_f})", use_container_width=True)
                    if landing_img is not None:
                        iv_col2.image(landing_img, caption=f"AI Detected Landing (Frame {l_nd_f})", use_container_width=True)
                    
                    cap_verify.release()
                    
                    # Graph
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(valid_frames, y_smooth, color="#2ecc71")
                    ax.axvspan(t_off_f, l_nd_f, color='yellow', alpha=0.3)
                    st.pyplot(fig)
                else: st.warning("No jump detected.")
            else: st.error("Tracking failed.")

    cap.release()
