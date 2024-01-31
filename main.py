import streamlit as st
import cv2
import numpy as np
import torch
import re

import easyocr

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

plat_date_any = False
plat_num_any = False

trigger_executed = False


def trigger_image_on_center(detected_image, frame):
    # Use the global keyword to modify the global variable
    global trigger_executed
    global plat_date_any
    global plat_num_any

    horizontal_line_y = frame.shape[0] // 2 + 24
        
    cv2.line(detected_image, (0, horizontal_line_y), (frame.shape[1], frame.shape[0] // 2 + 22), (0, 255, 0), 2)

    if not trigger_executed:
        for det in results.xyxy[0]:
            if det[-1] == 0: 
                bbox = det[:4].cpu().numpy().astype(int)
                center_y = (bbox[1] + bbox[3]) // 2

                cv2.rectangle(detected_image, (bbox[0], center_y - 5), (bbox[2], center_y + 5), (255, 0, 0), 5)

                if horizontal_line_y - 5 <= center_y <= horizontal_line_y + 5:
                    zoom_and_save_image(results.xyxy[0].cpu().numpy(), frame)

                

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
            detect_plat_information()


        except Exception as e:
            print(f"Error processing object: {e}")
            continue
    
def save_plat_info_txt(text):
    with open('database/data.txt', 'a') as file:
        file.write('text\n')

def detect_plat_information():
    img = "output/cropped/temp.png"

    results = cropped_model(img)
    boxes = results.xyxy[0].cpu().numpy()

    # Define labels for platNum and platDate
    target_labels = ["platNum", "PlatDate"]

    for box in boxes:
        try:
            x1, y1, x2, y2, confidence, class_idx = box
            detected_label = cropped_model.names[int(class_idx)]

            print(f"Detected label: {detected_label}")

            # Check if the detected label matches platNum or platDate
            if any(target_label in detected_label for target_label in target_labels):
                # Calculate the width and height of the bounding box
                width = x2 - x1
                height = y2 - y1

                # Crop the object from the original image
                cropped_object = cv2.imread(img)
                cropped_object = cropped_object[int(y1):int(y2), int(x1):int(x2)]
                cropped_object_bw = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2GRAY)
                # Increase the contrast
                cropped_object_high_contrast = cv2.equalizeHist(cropped_object_bw)

                read_plat_information(detected_label, cropped_object)
            
        except Exception as e:
            print(f"Error processing object: {e}")
            continue



def read_plat_information(detected_label, img):
    # Use the global keyword to modify the global variable
    global trigger_executed
    global plat_date_any
    global plat_num_any

    reader = easyocr.Reader(['en'])

    result = reader.readtext(img)
    # Tampilkan hasil OCR
    for detection in result:
        text = detection[1]
        text_items.append(f"{detected_label} : {text}")
        text_list.write("\n".join(text_items))

        if detected_label == "platNum" and text != "":
            plat_num_any = True
        elif detected_label == "PlatDate" and text != "":
            plat_date_any = True

            flag = 0
            numbers = re.findall(r'\d+', text)
            if len(numbers) >= 2:
                number1 = int(numbers[0])
                number2 = int(numbers[1])

                if int(number2) > 22:
                    st.success("Plat Nomor Masih Aktif")
                    text_list.write("\n".join(text_items))
                elif int(number2) == 22:
                    if int(number1) > 11:
                        st.success("Plat Nomor Masih Aktif")
                        text_list.write("\n".join(text_items))
                    else:
                        text_list.write("\n".join(text_items))
                        st.error("Plat Nomor Sudah Tidak Aktif")
                else:
                    text_list.write("\n".join(text_items))
                    st.error("Plat Nomor Sudah Tidak Aktif")
            else:
                plat_date_any = False
                flag += 1
                if flag == 2:
                    if int(text) > 23:
                        st.success("Plat Nomor Masih Aktif")
                        text_list.write("\n".join(text_items))
                    else:
                        text_list.write("\n".join(text_items))
                        st.error("Plat Nomor Sudah Tidak Aktif")

                

        if plat_num_any and plat_date_any:
            trigger_executed = True



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

    cap = cv2.VideoCapture("test/test.mp4")
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







