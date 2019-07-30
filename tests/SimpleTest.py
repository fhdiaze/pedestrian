import numpy as np
from pedestrian.detection.YoloV3Voc import YoloV3Voc
from pedestrian.detection.YoloV3Coco import YoloV3Coco
from pedestrian.position.TwoCornersPM import TwoCornersPM
from PIL import Image, ImageDraw


detector = YoloV3Coco()
img = Image.open("/home/investigacion/Pictures/transmi2.png")

# Variables
width, height = img.size
in_size = np.array([416, 416])
s_range = np.array([[0.0, width], [0.0, height]])
t_range = np.array([[0.0, 1.0], [0.0, 1.0]])

in_img = img.resize(in_size)
in_tensor = np.array(in_img, dtype=np.float32).reshape(-1, in_size[0], in_size[1], 3)

boxes = detector.detect(in_tensor)
pm = TwoCornersPM()
pm.scale()
print(np.array(boxes).shape)

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(box)

img.show()
img.save("/home/investigacion/Pictures/transmi2BoxCoco.png")

print(boxes)