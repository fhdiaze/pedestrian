import numpy as np
import os
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

with Image.open(os.path.join(in_path, in_name)) as img:

    # Variables
    width, height = img.size
    s_range = np.array([[0.0, detector.in_size[0]], [0.0, detector.in_size[1]]])
    t_range = np.array([[0.0, width], [0.0, height]])
    in_tensor = np.array(img, dtype=np.float32)

    boxes = detector.detect(in_tensor)

    if boxes.size != 0:
        for box in boxes[:, :4]:
            pm.plot(img, list(box), outline)

        img.save(os.path.join(out_path, out_name))
