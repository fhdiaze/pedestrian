import tensorflow as tf
import tensornets as nets
import numpy as np


class YoloV3Voc(object):
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        self.model = nets.YOLOv3VOC(self.inputs)
        self.vocPersonClass = 14

    def detect(self, frame):
        """

        :param np.darray(416, 416, 3) frame:
        :return:
        """
        with tf.Session() as sess:
            sess.run(self.model.pretrained())
            preds = sess.run(self.model.preds, {self.inputs: self.model.preprocess(frame)})
            boxes = self.model.get_boxes(preds, frame.shape[1:3])
            boxes = np.array(boxes[self.vocPersonClass])

            if boxes.size != 0:
                boxes = boxes[:, :4]

        return boxes
