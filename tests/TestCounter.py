import os
import cv2
import numpy as np
from pedestrian.counting.Counter import Counter
from pedestrian.detection.MobileSSD import MobileSSD
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
counter = Counter(detector, det_period, tracker)

# Environment Variables
# in_path = "/home/investigacion/Downloads/"
in_path = "C:/Users/kuby/Documents/Fredy/Input/"
# out_path = "/home/investigacion/Downloads/"
out_path = "C:/Users/kuby/Documents/Fredy/Output/"
in_name = "example_01.mp4"
in_video = os.path.join(in_path, in_name)

# Loading Video
cap = cv2.VideoCapture(os.path.join(in_path, in_name))

f = 0
while cap.isOpened() and f < 1000:
    ret, frame = cap.read()  # BGR code

    if ret:
        counter.update(frame)

    f += 1

print(counter.up, counter.down)
# When everything done, release the video capture and video write objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
