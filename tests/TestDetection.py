import numpy as np
import os
import cv2
from pedestrian.detection.YoloV3Voc import YoloV3Voc
from pedestrian.detection.YoloV3Coco import YoloV3Coco
from pedestrian.detection.MobileSSD import MobileSSD
from pedestrian.position.TwoCornersPM import TwoCornersPM
from PIL import Image, ImageDraw

proto = "C:/Users/kuby/Downloads/object-detection-deep-learning/MobileNetSSD_deploy.prototxt.txt"
model = "C:/Users/kuby/Downloads/object-detection-deep-learning/MobileNetSSD_deploy.caffemodel"
confidence = 0.2
detector = MobileSSD(proto, model, confidence)
det_name = type(detector).__name__
#in_path = "/home/investigacion/Pictures/InputImages"
in_path = "C:/Users/kuby/Downloads/"
out_path = "/home/investigacion/Pictures/OutputImages"
out_path = "C:/Users/kuby/Downloads/"
in_name = "TM_evasores_00127.jpg"
out_name = det_name + "Box" + in_name
outline = "blue"
pm = TwoCornersPM()

frame = cv2.imread(os.path.join(in_path, in_name))
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
boxes = detector.detect(rgb_frame)

if boxes.size != 0:
    for box in boxes[:, :4]:
        pm.plot(frame, box.astype("int"), (255, 0, 0))

    cv2.imwrite(os.path.join(out_path, out_name), frame)
