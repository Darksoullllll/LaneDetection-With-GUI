import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from PIL import Image
import io

class FindLaneLines:
    def __init__(self):
        """ Init Application"""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

# Define a function to process the uploaded video file
def process_video_file(video_file):
    """ Process video file """
    with open(video_file.name, "wb") as f:
        f.write(video_file.getbuffer())

    findLaneLines = FindLaneLines()
    findLaneLines.process_video(video_file.name, 'result.mp4')

def process_img_file(img_file):
    '''img = Image.open(img_file)
    img_buffer = io.BytesIO()
    img.save(img_buffer,format="JPEG")
    with open(img_file,"wb") as f1:
        f1.write(img_buffer.getbuffer())'''
    findLaneLines = FindLaneLines()
    findLaneLines.process_image(img_file.name,'res.jpg') 

def main():
    st.title("LANE DETECTION")
    uploaded_file = st.file_uploader("Choose a video file You want to Detect the Lane", type=["mp4"])
    img_upload = st.file_uploader("Choose a picture you want to detect the lane",type=["jpg","jpeg"])
    if uploaded_file is not None:
        process_video_file(uploaded_file)

        st.video('result.mp4')
    if img_upload is not None:
        process_img_file(img_upload)

if __name__ == '__main__':
    main()
