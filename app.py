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
    tab1, tab2 = st.tabs(["Image", "Video"])

    with tab1:
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
    
    with tab2:
        vid_file = st.file_uploader("Video", type=['mp4'])

        if vid_file is not None:
            stframe = st.empty()
            with open("out.mp4","wb") as f:
                f.write(vid_file.getbuffer())
                
            st.write('Violence Detection')
            results = viol("out.mp4", stream=True, verbose=False, conf=0.5)

            for result in results:
                stframe.image(result.plot(), channels="BGR", width=700)


elif file_type == "Fall Detection":
    tab1, tab2 = st.tabs(["Image", "Video"])

    with tab1:
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
    
    with tab2:
        vid_file = st.file_uploader("Video", type=['mp4'])

        if vid_file is not None:
            stframe = st.empty()
            with open("out.mp4","wb") as f:
                f.write(vid_file.getbuffer())
                
            st.write('Fire and Smoke Detection')
            results = fall("out.mp4", stream=True, verbose=False, conf=0.5)

            for result in results:
                stframe.image(result.plot(), channels="BGR", width=700)



elif file_type == "Fire and Smoke Detection":
    tab1, tab2 = st.tabs(["Image", "Video"])

    with tab1:
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
    
    with tab2:
        vid_file = st.file_uploader("Video", type=['mp4'])

        if vid_file is not None:
            stframe = st.empty()
            with open("out.mp4","wb") as f:
                f.write(vid_file.getbuffer())
                
            st.write('Fire and Smoke Detection')
            results = fire("out.mp4", stream=True, verbose=False, conf=0.5)

            for result in results:
                stframe.image(result.plot(), channels="BGR", width=700)


elif file_type == "Smoker Detection":
    tab1, tab2 = st.tabs(["Image", "Video"])

    with tab1:
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
    
    with tab2:
        vid_file = st.file_uploader("Video", type=['mp4'])

        if vid_file is not None:
            stframe = st.empty()
            with open("out.mp4","wb") as f:
                f.write(vid_file.getbuffer())
                
            st.write('Crash Detection')
            results = smoker("out.mp4", stream=True, verbose=False, conf=0.5)

            for result in results:
                stframe.image(result.plot(), channels="BGR", width=700)


    # image_file = st.file_uploader("Video", type=['mp4'])


    # if image_file is not None:
    #     stframe = st.empty()
    #     with open("out.mp4","wb") as f:
    #         f.write(image_file.getbuffer())
            
    #     vf = cv2.VideoCapture("out.mp4")
        
    #     st.write('Last detected face')
        # Create grid columns for displaying detected faces
        
        # while vf.isOpened():
        #     success, frame = vf.read()

        #     if success:
        #         frc = frame.copy()
        #         # Detect and display faces in the video frames
        #         for box in fire.track(frame, persist=True, verbose=False)[0].boxes:
        #             if box.conf < 0.5:
        #                 continue

        #             x1, y1, x2, y2 = map(int, box.xyxy[0])
        #             id = int(box.id) if box.id else 0

        #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        #             cv2.putText(frame, f"id: {id}", (int(x1), int(y1) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        #                         (255, 255, 255), 2)

        #         stframe.image(frame, channels="BGR", width=700)
        #     else:
        #         break
        # results = fire(source=0, stream=True, conf=0.8)  # return a generator of Results objects

        # # Process results generator
        # for result in results:
        #     stframe.image(result.plot(), channels="BGR", width=700)

        # image = Image.open(image_file)
        # st.image(image, caption='Uploaded image', use_column_width=True)

        # with open("out.jpg","wb") as f:
        #     f.write(image_file.getbuffer())
        # shutil.rmtree('runs', ignore_errors=True)
        # fire("out.jpg", verbose=False, save=True)

        # st.title("Predicted")
        # st.image(Image.open('runs/detect/predict/out.jpg'), caption='Predicted image', use_column_width=True)
        # shutil.rmtree('runs', ignore_errors=True)


# else:
#     image_file = st.file_uploader("Image", type=['jpg', 'jpeg', 'png'])

#     if image_file is not None:
#         image = Image.open(image_file)
#         st.image(image, caption='Uploaded image', use_column_width=True)

#         with open("out.jpg","wb") as f:
#             f.write(image_file.getbuffer())

#         HazeImg = cv2.imread('out.jpg')
#         HazeCorrectedImg, HazeMap = image_dehazer.remove_haze(HazeImg)

#         st.title("Predicted")
#         st.image(HazeCorrectedImg[:,:,::-1], caption='Predicted image', use_column_width=True)