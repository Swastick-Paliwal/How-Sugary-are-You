import streamlit as st
import cv2
import numpy as np
import time
import config

# Page setup
st.set_page_config(page_title="Sugahhhh", layout="centered")
st.title("How 'Sugary' are You 😏")
st.caption("Upload your pics or use the webcam to get an AI-generated attractiveness score!")

# ---------------------------
# Gender selection
# ---------------------------
config.gender = st.radio("Select your gender:", ["Male", "Female"], horizontal=True)

# ---------------------------
# Helper Functions
# ---------------------------
def capture_from_webcam_live_preview(countdown_text="Capturing"):
    stframe = st.empty()  # Placeholder for the webcam feed
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    duration = 5  # Countdown duration in seconds
    final_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            cap.release()
            return None

        elapsed = int(time.time() - start_time)
        remaining = duration - elapsed

        if remaining <= 0:
            final_frame = frame
            break

        # Overlay countdown text
        overlay = frame.copy()
        cv2.putText(overlay, f"{countdown_text} in {remaining}...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # Show the frame in Streamlit
        stframe.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Control FPS (sleep for small amount)
        time.sleep(0.05)  # ~20 FPS

    cap.release()
    st.success("📸 Image captured!")
    return final_frame


def read_image(file_bytes):
    img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image

# ---------------------------
# Face image upload/capture
# ---------------------------
st.subheader("Capture or Upload Your Face Image")
face_col1, face_col2 = st.columns(2)

with face_col1:
    uploaded_face = st.file_uploader("Upload a face image", key="face", type=["jpg", "jpeg", "png"])
with face_col2:
    if st.button("📸 Capture Face from Webcam"):
        config.face_image = capture_from_webcam_live_preview("Capturing FACE")

if uploaded_face is not None:
    config.face_image = read_image(uploaded_face.read())

if config.face_image is not None:
    st.image(cv2.cvtColor(config.face_image, cv2.COLOR_BGR2RGB), caption="Face Image", use_container_width=True)

# ---------------------------
# Body image upload/capture
# ---------------------------
st.subheader("Capture or Upload Your Body Image")
body_col1, body_col2 = st.columns(2)

with body_col1:
    uploaded_body = st.file_uploader("Upload a body image", key="body", type=["jpg", "jpeg", "png"])
with body_col2:
    if st.button("📸 Capture Body from Webcam"):
        config.body_image = capture_from_webcam_live_preview("Capturing BODY")

if uploaded_body is not None:
    config.body_image = read_image(uploaded_body.read())

if config.body_image is not None:
    st.image(cv2.cvtColor(config.body_image, cv2.COLOR_BGR2RGB), caption="Body Image", use_container_width=True)

# ---------------------------
# Placeholder for results (to be integrated)
# ---------------------------
if config.face_image is not None and config.body_image is not None:
    st.success("Both images captured successfully!")
    st.write(f"Selected Gender: **{config.gender}**")

    # MOCK CALL TO TEAMMATE FUNCTIONS (replace later)
    # from face_model import get_face_score
    # from body_model import get_body_score
    #
    # face_attractiveness = get_face_score(config.face_image)
    # body_attractiveness = get_body_score(config.body_image)
    #
    # st.metric("Face Score", f"{face_attractiveness}%")
    # st.metric("Body Score", f"{body_attractiveness}%")
    # overall = (face_attractiveness + body_attractiveness) // 2
    # st.title(f"💯 Overall Attractiveness: {overall}%")