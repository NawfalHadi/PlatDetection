import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set the page configuration
st.set_page_config(
    page_title="Camera and Text Streamlit App",
    page_icon="📷",
    layout="wide",  # Set the layout to wide
)

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

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()

        if not ret:
            st.write("The video capture has ended.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

        # Add new text to the text_list on each iteration
        new_text = f"New text: {np.random.randint(1, 100)}"
        text_items.append(new_text)
        text_list.write("\n".join(text_items))

        if stop_button_pressed:
            break

    cap.release()
