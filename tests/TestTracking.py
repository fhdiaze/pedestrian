import os
import cv2
import numpy as np
from PIL import Image
from PIL import ImageColor

from pedestrian.detection.MobileSSD import MobileSSD
from pedestrian.detection.YoloV3Voc import YoloV3Voc
from pedestrian.detection.YoloV3Coco import YoloV3Coco
from pedestrian.position.TwoCornersPM import TwoCornersPM
from pedestrian.tracking.Sort import Sort

# Pipeline Variables
det_period = 1
outline = "blue"
proto = "C:/Users/kuby/Downloads/object-detection-deep-learning/MobileNetSSD_deploy.prototxt.txt"
model = "C:/Users/kuby/Downloads/object-detection-deep-learning/MobileNetSSD_deploy.caffemodel"
confidence = 0.2
detector = MobileSSD(proto, model, confidence)
pm = TwoCornersPM()
tracker = Sort()

# Environment Variables
# in_path = "/home/investigacion/Downloads/"
in_path = "C:/Users/kuby/Downloads/Input/"
# out_path = "/home/investigacion/Downloads/"
out_path = "C:/Users/kuby/Downloads/Output/"
in_name = "Ch2_20181110121206.mp4"
out_name = "Box_" + os.path.splitext(in_name)[0] + ".avi"
in_video = os.path.join(in_path, in_name)
out_video = os.path.join(out_path, out_name)
out_fps = 30

COLORS = np.random.uniform(0, 255, size=(100, 3))

# Loading Video
cap = cv2.VideoCapture(os.path.join(in_path, in_name))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
t_range = np.array([[0.0, video_width], [0.0, video_height]])
out = cv2.VideoWriter(out_video, cv2. VideoWriter_fourcc(*"MJPG"), out_fps, (video_width, video_height))

f = 0
while cap.isOpened() and f < 1000:
    ret, bgr_frame = cap.read()

    if ret:
        boxes = np.empty((0, 0))

        if f % det_period == 0:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            boxes = detector.detect(rgb_frame)

        tracks = tracker.track(boxes.astype("int"))

        for track in tracks:
            pm.plot(bgr_frame, track[:-1].astype("int"), COLORS[1])

        out.write(bgr_frame)

    f += 1

# When everything done, release the video capture and video write objects
out.release()
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
