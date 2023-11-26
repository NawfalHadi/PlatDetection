import streamlit as st
import cv2
import numpy as np
import torch

from PIL import Image

# Set the page configuration
st.set_page_config(
    page_title="Camera and Text Streamlit App",
    page_icon="📷",
    layout="wide",  # Set the layout to wide
)

# Function to load the YOLOv5 models
@st.cache_resource()
def load_models():
    plat_model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/plat/best.pt')
    cropped_model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/information/best.pt')
    return plat_model, cropped_model

# Load YOLOv5 models outside the main Streamlit loop
plat_model, cropped_model = load_models()

# Create a two-column layout for the app
col1, col2 = st.columns(2)

# In the right column, list text or other content
with col1:
    st.header("Text List")

    # List to store text
    text_items = []

    # Placeholder for displaying text
    text_list = st.empty()

# In the left column, capture video from the camera
with col2:
    st.header("Camera Feed")

    cap = cv2.VideoCapture("test/video.mp4")
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()

        if not ret:
            st.write("The video capture has ended.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = plat_model(frame)
        results.print()
        image_data = results.ims[0]

        pil_image = Image.fromarray(image_data.astype("uint8"))

        frame_placeholder.image(pil_image, channels="RGB")

        # # Add new text to the text_list on each iteration
        # new_text = f"New text: {np.random.randint(1, 100)}"
        # text_items.append(new_text)
        # text_list.write("\n".join(text_items))

        if stop_button_pressed:
            break

    cap.release()

