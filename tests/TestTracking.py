import numpy as np
import os
import cv2
# from pedestrian.detection.YoloV3Voc import YoloV3Voc
# from pedestrian.detection.YoloV3Coco import YoloV3Coco
from pedestrian.position.TwoCornersPM import TwoCornersPM
from pedestrian.tracking.Sort import Sort
from PIL import Image, ImageDraw

# VARS
# detector = YoloV3Coco()
tracker = Sort()
#det_name = type(detector).__name__
in_path = "/home/investigacion/Pictures/InputImages"
out_path = "/home/investigacion/Pictures/OutputImages"
img_name = "TM_evasores_00008.jpg"
#out_img_name = det_name + "Box" + img_name
outline = "blue"
in_size = np.array([416, 416])
s_range = np.array([[0.0, in_size[0]], [0.0, in_size[1]]])

# Video processing
cap = cv2.VideoCapture("C:/Users/kuby/Downloads/Ch4_20181029060955.mp4")

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
size = (width, height)
t_range = np.array([[0.0, width], [0.0, height]])
tracks = []
out = cv2.VideoWriter(filename="C:/Users/kuby/Downloads/Ch4_out.mp4", apiPreference=cv2.CAP_GSTREAMER, fourcc=cv2.VideoWriter_fourcc(*"DIVX"), fps=15, frameSize=size)
f = 0
while cap.isOpened() and f < 300:
    ret, frame = cap.read()
    f += 1
    if ret:
        img = Image.fromarray(frame)
        in_img = img.resize(in_size)
        in_tensor = np.array(in_img, dtype=np.float32).reshape(-1, in_size[0], in_size[1], 3)
        # dets = detector.detect(frame)
        # tracks = tracker.update(dets)

        # boxes = detector.detect(in_tensor)
        pm = TwoCornersPM()
        #boxes = pm.scale(boxes, s_range, t_range)

        #for box in boxes:
        #    pm.plot(img, list(box), outline)

        out.write(np.array(img, dtype=np.float32))

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()