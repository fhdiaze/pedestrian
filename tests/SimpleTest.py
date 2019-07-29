import numpy as np
from pedestrian.detection.YoloV3Voc import YoloV3Voc
from pedestrian.detection.YoloV3Coco import YoloV3Coco
from PIL import Image, ImageDraw

detector = YoloV3Coco()
img = Image.open("/home/investigacion/Pictures/transmi2.png")
img = img.resize((416,416))
ima = np.array(img, dtype=np.float32).reshape(-1, 416, 416, 3)

boxes = detector.detect(ima)
print(np.array(boxes).shape)

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(box)

img.show()
img.save("/home/investigacion/Pictures/transmi2BoxCoco.png")

print(boxes)