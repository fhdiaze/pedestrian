import numpy as np
import os
import cv2

from pedestrian.detection.HOGDetector import HOGDetector
from pedestrian.detection.YoloV3Voc import YoloV3Voc
from pedestrian.detection.YoloV3Coco import YoloV3Coco
from pedestrian.detection.MobileSSD import MobileSSD
from pedestrian.position.TwoCornersPM import TwoCornersPM

# Pipeline Variables
workspace = "C:/Users/kuby/Documents/Fredy/Workspace"
# workspace = "/home/investigacion/Documents/Workspace"
proto = os.path.join(workspace, "object-detection-deep-learning/MobileNetSSD_deploy.prototxt.txt")
model = os.path.join(workspace, "object-detection-deep-learning/MobileNetSSD_deploy.caffemodel")
confidence = 0.2
# detector = MobileSSD(proto, model, confidence)
# detector = YoloV3Coco()
detector = HOGDetector(confidence=confidence)
det_name = type(detector).__name__

# Environment Variables
in_path = os.path.join(workspace, "Input")
out_path = os.path.join(workspace, "Output")
in_name = "person_010.bmp"
out_name = det_name + "Box" + in_name
outline = "blue"
pm = TwoCornersPM()

frame = cv2.imread(os.path.join(in_path, in_name))
boxes = detector.detect(frame)

if boxes.size != 0:
    for box in boxes[:, :4]:
        pm.plot(frame, box.astype("int"), (255, 0, 0))

    cv2.imwrite(os.path.join(out_path, out_name), frame)
