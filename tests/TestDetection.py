import numpy as np
import os
from pedestrian.detection.YoloV3Voc import YoloV3Voc
from pedestrian.detection.YoloV3Coco import YoloV3Coco
from pedestrian.position.TwoCornersPM import TwoCornersPM
from PIL import Image, ImageDraw


detector = YoloV3Voc()
det_name = type(detector).__name__
in_path = "/home/investigacion/Pictures/InputImages"
out_path = "/home/investigacion/Pictures/OutputImages"
img_name = "transmi.jpg"
out_img_name = det_name + "Box" + img_name
outline = "blue"

with Image.open(os.path.join(in_path, img_name)) as img:

    # Variables
    width, height = img.size
    in_size = np.array([416, 416])
    s_range = np.array([[0.0, in_size[0]], [0.0, in_size[1]]])
    t_range = np.array([[0.0, width], [0.0, height]])

    in_img = img.resize(in_size)
    in_tensor = np.array(in_img, dtype=np.float32).reshape(-1, in_size[0], in_size[1], 3)

    boxes = detector.detect(in_tensor)

    if boxes.size != 0:
        pm = TwoCornersPM()
        boxes[:, :4] = pm.scale(boxes[:, :4], s_range, t_range)

        for box in boxes[:, :4]:
            pm.plot(img, list(box), outline)

        img.save(os.path.join(out_path, out_img_name))
