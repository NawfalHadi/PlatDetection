import streamlit as st
import cv2
import numpy as np
import torch

from PIL import Image
from matplotlib import pyplot as plt


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

def trigger_image_on_center(detected_image, frame):
    horizontal_line_y = frame.shape[0] // 2 + 24
        
    cv2.line(detected_image, (0, horizontal_line_y), (frame.shape[1], frame.shape[0] // 2 + 22), (0, 255, 0), 2)

    for det in results.xyxy[0]:
        if det[-1] == 0: 
            bbox = det[:4].cpu().numpy().astype(int)
            center_y = (bbox[1] + bbox[3]) // 2

            cv2.rectangle(detected_image, (bbox[0], center_y - 5), (bbox[2], center_y + 5), (255, 0, 0), 2)

            if horizontal_line_y - 5 <= center_y <= horizontal_line_y + 5:
                    
                zoom_and_save_image(results.xyxy[0].cpu().numpy(), frame)
                detect_plat_information()

                text_items.append("License Plate Detected: Got it!")
                text_list.write("\n".join(text_items))

def zoom_and_save_image(boxes, img):
    i = 0

    for box in boxes:
        i += 1
        try:
            x1, y1, x2, y2, confidence, class_idx = box
            # Calculate the width and height of the bounding box
            width = x2 - x1
            height = y2 - y1

            # Determine the center of the bounding box
            center_x = x1 + width / 2
            center_y = y1 + height / 2

            # Define a desired width and height for the cropped object
            desired_width = 600  # Adjust this value to your preference
            desired_height = 200  # Adjust this value to your preference

            # Calculate the new coordinates for cropping
            new_x1 = int(center_x - desired_width / 2)
            new_y1 = int(center_y - desired_height / 2)
            new_x2 = int(center_x + desired_width / 2)
            new_y2 = int(center_y + desired_height / 2)

            # Crop the object from the original image
            cropped_object = img[new_y1:new_y2, new_x1:new_x2]
            
            
            # Display or save the cropped object
            plt.imshow(cropped_object)
            plt.show()

            # Turn off axis labels and ticks
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])

            # Save the image
            plt.savefig(f'output/cropped/temp.png', bbox_inches='tight', pad_inches=0)

        except Exception as e:
            print(f"Error processing object: {e}")
            continue
    

def detect_plat_information():
    img = "output/cropped/temp.png"

    results = cropped_model(img)
    results.print()


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

        # Y-coordinate of the horizontal line
        
        results = plat_model(frame)
        detected_image = frame.copy()

        trigger_image_on_center(detected_image, frame)

        # image_data = results.ims[0]

        pil_image = Image.fromarray(detected_image.astype("uint8"))

        frame_placeholder.image(pil_image, channels="RGB")

        # # Add new text to the text_list on each iteration
        # new_text = f"New text: {np.random.randint(1, 100)}"
        # text_items.append(new_text)
        # text_list.write("\n".join(text_items))

        if stop_button_pressed:
            break

    cap.release()
    print("finish")



def read_plat_information():
    pass

def save_plat_info_txt():
    pass

