import numpy as np
import os
import cv2
# from pedestrian.detection.YoloV3Voc import YoloV3Voc
# from pedestrian.detection.YoloV3Coco import YoloV3Coco
from pedestrian.position.TwoCornersPM import TwoCornersPM
from pedestrian.tracking.Sort import Sort
from PIL import Image, ImageDraw

# Pipeline Variables
in_size = (416, 416)
s_range = np.array([[0.0, in_size[0]], [0.0, in_size[1]]])
outline = "blue"
# detector = YoloV3Coco()
pm = TwoCornersPM()
tracker = Sort()

# Environment Variables
in_path = "C:/Users/kuby/Downloads"
out_path = "C:/Users/kuby/Downloads"
in_name = "Ch4_20181029060955.mp4"
out_name = "Box_" + in_name
in_video = os.path.join(in_path, in_name)
out_video = os.path.join(out_path, out_name)

# Loading Video
cap = cv2.VideoCapture(os.path.join(in_path, in_name))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
t_range = np.array([[0.0, video_width], [0.0, video_height]])

with cv2.VideoWriter(in_video, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"DIVX"), 15, (video_width, video_height)) as out:
    f = 0
    while cap.isOpened() and f < 3:
        ret, frame = cap.read()
        f += 1
        if ret:
            img = Image.fromarray(frame)
            in_img = img.resize(in_size)
            in_tensor = np.array(in_img, dtype=np.float32).reshape(-1, in_size[0], in_size[1], 3)
            # boxes = detector.detect(frame)
            # boxes[:, :4] = pm.scale(boxes[:, :4], s_range, t_range)
            # tracker.update(boxes)
            # tracks = tracker.predict()

            #for track in tracks:
            #    pm.plot(img, list(track), outline)

            out.write(np.array(img, dtype=np.float32))

    # When everything done, release the video capture and video write objects
    out.release()
    cap.release()

# Closes all the frames
cv2.destroyAllWindows()