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
workspace = "/home/investigacion/Documents/Workspace"
# workspace = "/home/investigacion/Documents/Workspace"
det_period = 5
max_unseen = 1
out_fps = 30
proto = os.path.join(workspace, "object-detection-deep-learning/MobileNetSSD_deploy.prototxt.txt")
model = os.path.join(workspace, "object-detection-deep-learning/MobileNetSSD_deploy.caffemodel")
confidence = 0.1
detector = MobileSSD(proto, model, confidence)
pm = TwoCornersPM()
connector = DistanceConnector(200)
tracker = Centroid(connector, detector, det_period, max_unseen)
counter = Counter(tracker)

# Environment Variables
in_path = os.path.join(workspace, "Input")
out_path = os.path.join(workspace, "Output")
in_name = "example_01.mp4"
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
while cap.isOpened() and ret and f < 2000:
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
