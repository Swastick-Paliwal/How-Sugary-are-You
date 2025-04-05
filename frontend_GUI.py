import streamlit as st
import cv2
import numpy as np
import time
import config
import base64
import face

# Page setup
st.set_page_config(page_title="Sugahhhh", layout="centered")
st.title("How 'Sugary' are You 😏")
st.caption("Upload your pics or use the webcam to get an AI-generated attractiveness score!")

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def place_logo():
    logo_data = get_base64("logo.png")  # Change to your image file
    st.markdown(f"""
        <style>
        .custom-logo {{
            position: fixed;
            top: 15px;
            left: 20px;
            z-index: 1000;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0px 0px 8px rgba(255, 255, 255, 0.3);
        }}
        </style>
        <img src="data:image/png;base64,{logo_data}" class="custom-logo">
    """, unsafe_allow_html=True)

place_logo()

st.markdown("""
    <style>
    /* Hide Streamlit's default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: rgb(245, 132, 188);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
    }
    div.stButton > button:first-child:hover {
        background-color: deeppink;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #2b1055, #7597de);
        background-attachment: fixed;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
# Happy Face image upload/capture
# ---------------------------
st.subheader("Capture or Upload Your Happy Face Image")
face_col1, face_col2 = st.columns(2)

with face_col1:
    uploaded_happy_face = st.file_uploader("Upload a happy face image", key="happy_face", type=["jpg", "jpeg", "png"])
with face_col2:
    if st.button("📸 Capture from Webcam", key="capture_happy"):
        config.happy_face_image = capture_from_webcam_live_preview("Capturing FACE")

if uploaded_happy_face is not None:
    config.happy_face_image = read_image(uploaded_happy_face.read())

if config.happy_face_image is not None:
    st.image(cv2.cvtColor(config.happy_face_image, cv2.COLOR_BGR2RGB), caption="Happy Face Image", use_container_width=True)

# ---------------------------
# Serious Face image upload/capture
# ---------------------------
st.subheader("Capture or Upload Your Serious Face Image")
face_col1, face_col2 = st.columns(2)

with face_col1:
    uploaded_serious_face = st.file_uploader("Upload a serious face image", key="serious_face", type=["jpg", "jpeg", "png"])
with face_col2:
    if st.button("📸 Capture from Webcam", key="capture_serious"):
        config.serious_face_image = capture_from_webcam_live_preview("Capturing FACE")

if uploaded_serious_face is not None:
    config.serious_face_image = read_image(uploaded_serious_face.read())

if config.serious_face_image is not None:
    st.image(cv2.cvtColor(config.serious_face_image, cv2.COLOR_BGR2RGB), caption="Face Image", use_container_width=True)

# ---------------------------
# Body image upload/capture
# ---------------------------
st.subheader("Capture or Upload Your Body Image")
body_col1, body_col2 = st.columns(2)

with body_col1:
    uploaded_body = st.file_uploader("Upload a body image", key="body", type=["jpg", "jpeg", "png"])
with body_col2:
    if st.button("📸 Capture from Webcam", key="capture_body"):
        config.body_image = capture_from_webcam_live_preview("Capturing BODY")

if uploaded_body is not None:
    config.body_image = read_image(uploaded_body.read())

if config.body_image is not None:
    st.image(cv2.cvtColor(config.body_image, cv2.COLOR_BGR2RGB), caption="Body Image", use_container_width=True)

# ---------------------------
# Placeholder for results (to be integrated)
# ---------------------------
if config.happy_face_image is not None and config.happy_face_image is not None and config.body_image is not None:
    face.main()
    st.success("All three images captured successfully!")
    st.write(f"Selected Gender: **{config.gender}**")

    st.write(f"Emotion: {config.emotion_score}/n Symmetry: {config.symmetry_score}")
    # body_attractiveness = get_body_score(config.body_image)
    #
    # st.metric("Face Score", f"{face_attractiveness}%")
    # st.metric("Body Score", f"{body_attractiveness}%")
    # overall = (face_attractiveness + body_attractiveness) // 2
    # st.title(f"Overall Attractiveness: {overall}%")