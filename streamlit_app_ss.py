import SessionState as SessionState
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from streamlit_webrtc import webrtc_streamer
import av

#---------------------------------------------------------------------------------------
# The SessionState class allows us to initialize and save variables across for across
# session states. This is a valuable feature that enables us to take different actions
# depending on the state of selected variables in the code. If this is not done then
# all variables are reset any time the application state changes (e.g., when a user
# interacts with a widget). For example, the confidence threshold of the slider has
# changed, but we are still working with the same image, we can detect that by
# comparing the current file_uploaded_id (img_file_buffer.id) with the
# previous value (ss.file_uploaded_id) and if they are the same then we know we
# don't need to call the face detection model again. We just simply need to process
# the previous set of detections.
#---------------------------------------------------------------------------------------
USE_SS = True
if USE_SS:
    ss = SessionState.get(file_uploaded_id=-1, # Initialize file uploaded index.
                          detections=None)     # Initialize detections.

# Create application title and file uploader widget.
st.title("OpenCV Deep Learning based Face Detection")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# Function for detecting facses in an image.
def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections

# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            # Draw bounding boxes around detected faces.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes

# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


class VideoFaceProcessor:
    def recv(self, frame):
        frame1 = frame.to_ndarray(format = "bgr24")
        frame1 = cv2.flip(frame1, 1)
        return av.VideoFrame.from_ndarray(frame1, format = "bgr24")


net = load_model()

webrtc_streamer("Web Streamer", video_processor_factory=VideoFaceProcessor)
