import os
import cv2
import numpy as np
from PIL import Image
from PIL import ImageColor
from pedestrian.detection.YoloV3Voc import YoloV3Voc
from pedestrian.detection.YoloV3Coco import YoloV3Coco
from pedestrian.position.TwoCornersPM import TwoCornersPM
from pedestrian.tracking.Sort import Sort

# Pipeline Variables
in_size = (416, 416)
s_range = np.array([[0.0, in_size[0]], [0.0, in_size[1]]])
outline = "blue"
detector = YoloV3Voc()
pm = TwoCornersPM()
tracker = Sort()

# Environment Variables
# in_path = "/home/investigacion/Downloads/"
in_path = "C:/Users/kuby/Downloads/"
# out_path = "/home/investigacion/Downloads/"
out_path = "C:/Users/kuby/Downloads/"
in_name = "Ch1_20181118175157.mp4"
out_name = "Box_" + os.path.splitext(in_name)[0] + ".wmv"
in_video = os.path.join(in_path, in_name)
out_video = os.path.join(out_path, out_name)

# Loading Video
cap = cv2.VideoCapture(os.path.join(in_path, in_name))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
t_range = np.array([[0.0, video_width], [0.0, video_height]])
out = cv2.VideoWriter(out_video, cv2. VideoWriter_fourcc(*"MJPG"), 5, (video_width, video_height))

f = 0
while cap.isOpened() and f < 3:
    ret, frame = cap.read()
    f += 1
    if ret:
        img = Image.fromarray(frame)
        in_img = img.resize(in_size)
        in_tensor = np.array(in_img, dtype=np.float32).reshape(-1, in_size[0], in_size[1], 3)

        boxes = detector.detect(in_tensor)

        if boxes.size != 0:
            boxes[:, :4] = pm.scale(boxes[:, :4], s_range, t_range)

        tracks = tracker.update(boxes)

        for track in tracks:
            pm.plot(img, list(track[:-1]), list(ImageColor.colormap.keys())[int(track[-1])])

        out.write(np.array(img))

# When everything done, release the video capture and video write objects
out.release()
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
