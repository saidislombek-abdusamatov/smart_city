import streamlit as st
from PIL import Image
from ultralytics import YOLO
# import image_dehazer
import shutil
# import cv2


viol = YOLO('models/violence.pt')
fire = YOLO('models/fire.pt')
fall = YOLO('models/fall.pt')
smoker = YOLO('models/smoker.pt')

st.title("Smart City")

st.sidebar.title("Models")
file_type = st.sidebar.radio("", options=["Violence Detection", "Fall Detection", "Fire and Smoke Detection", "Smoker Detection"])

if file_type == "Violence Detection":
    image_file = st.file_uploader("Image", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded image', use_column_width=True)

        with open("out.jpg","wb") as f:
            f.write(image_file.getbuffer())
        
        shutil.rmtree('runs', ignore_errors=True)
        viol("out.jpg", verbose=False, save=True, conf=0.52)

        st.title("Predicted")
        st.image(Image.open('runs/detect/predict/out.jpg'), caption='Predicted image', use_column_width=True)
        shutil.rmtree('runs', ignore_errors=True)


elif file_type == "Fall Detection":

    image_file = st.file_uploader("Image", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded image', use_column_width=True)

        with open("out.jpg","wb") as f:
            f.write(image_file.getbuffer())
        
        shutil.rmtree('runs', ignore_errors=True)
        fall("out.jpg", verbose=False, save=True, conf=0.7)

        st.title("Predicted")
        st.image(Image.open('runs/detect/predict/out.jpg'), caption='Predicted image', use_column_width=True)
        shutil.rmtree('runs', ignore_errors=True)


elif file_type == "Fire and Smoke Detection":
    image_file = st.file_uploader("Image", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded image', use_column_width=True)

        with open("out.jpg","wb") as f:
            f.write(image_file.getbuffer())
        
        shutil.rmtree('runs', ignore_errors=True)
        fire("out.jpg", verbose=False, save=True, conf=0.5)

        st.title("Predicted")
        st.image(Image.open('runs/detect/predict/out.jpg'), caption='Predicted image', use_column_width=True)
        shutil.rmtree('runs', ignore_errors=True)

elif file_type == "Smoker Detection":
    image_file = st.file_uploader("Image", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption='Uploaded image', use_column_width=True)

        with open("out.jpg","wb") as f:
            f.write(image_file.getbuffer())
        
        shutil.rmtree('runs', ignore_errors=True)
        smoker("out.jpg", verbose=False, save=True, conf=0.5)

        st.title("Predicted")
        st.image(Image.open('runs/detect/predict/out.jpg'), caption='Predicted image', use_column_width=True)
        shutil.rmtree('runs', ignore_errors=True)
