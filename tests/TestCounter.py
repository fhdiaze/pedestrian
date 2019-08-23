import os
import cv2
import numpy as np
from pedestrian.counting.Counter import Counter
from pedestrian.detection.MobileSSD import MobileSSD
from pedestrian.position.TwoCornersPM import TwoCornersPM
from pedestrian.tracking.DistanceConnector import DistanceConnector
from pedestrian.tracking.multiple.Centroid import Centroid
from pedestrian.tracking.multiple.Sort import Sort

# Pipeline Variables
det_period = 5
out_fps = 30
proto = "C:/Users/kuby/Downloads/object-detection-deep-learning/MobileNetSSD_deploy.prototxt.txt"
model = "C:/Users/kuby/Downloads/object-detection-deep-learning/MobileNetSSD_deploy.caffemodel"
confidence = 0.2
detector = MobileSSD(proto, model, confidence)
pm = TwoCornersPM()
connector = DistanceConnector()
tracker = Centroid(connector)
counter = Counter(detector, det_period, tracker)

# Environment Variables
# in_path = "/home/investigacion/Downloads/"
in_path = "C:/Users/kuby/Documents/Fredy/Input/"
# out_path = "/home/investigacion/Downloads/"
out_path = "C:/Users/kuby/Documents/Fredy/Output/"
in_name = "example_02.mp4"
out_name = "Count_" + os.path.splitext(in_name)[0] + ".avi"
in_video = os.path.join(in_path, in_name)
out_video = os.path.join(out_path, out_name)

# Loading Video
cap = cv2.VideoCapture(os.path.join(in_path, in_name))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line = np.array([[0, video_height // 2], [video_width, video_height // 2]])
counter.line = line
out = cv2.VideoWriter(out_video, cv2. VideoWriter_fourcc(*"MJPG"), out_fps, (video_width, video_height))

f = 0
ret = True
while cap.isOpened() and ret and f < 1000:
    ret, frame = cap.read()  # BGR code

    if ret:
        counter.update(frame)
        out.write(frame)

    f += 1

print("UP: {}, DOWN: {}".format(counter.up, counter.down))
# When everything done, release the video capture and video write objects
out.release()
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
