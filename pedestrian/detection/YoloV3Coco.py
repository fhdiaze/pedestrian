import cv2
import tensorflow as tf
import tensornets as nets
import numpy as np
from pedestrian.detection.Detector import Detector


class YoloV3Coco(Detector):
    __slots__ = ["inputs", "model", "person_class", "in_size"]

    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        self.model = nets.YOLOv3COCO(self.inputs)
        self.person_class = 0
        self.in_size = (416, 416)

    def detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.resize(frame, self.in_size)
        frame = np.expand_dims(frame, axis=0)

        with tf.Session() as sess:
            sess.run(self.model.pretrained())
            preds = sess.run(self.model.preds, {self.inputs: self.model.preprocess(frame)})
            boxes = self.model.get_boxes(preds, frame.shape[1:3])
            boxes = np.array(boxes[self.person_class])

        return boxes
