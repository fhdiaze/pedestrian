import cv2
import numpy as np
from pedestrian.detection.Detector import Detector
from pedestrian.position.TwoCornersPM import TwoCornersPM


class MobileSSD(Detector):

    __slots__ = ["ssd", "in_size", "person_class", "confidence", "pm", "s_range"]

    def __init__(self, proto: str, model: str, confidence: float):
        self.ssd = cv2.dnn.readNetFromCaffe(proto, model)
        self.in_size = (300, 300)
        self.person_class = 15
        self.confidence = confidence
        self.pm = TwoCornersPM()
        self.s_range = np.array([[0.0, 1.0], [0.0, 1.0]])

    def detect(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, self.in_size), 0.007843, self.in_size, 127.5)
        self.ssd.setInput(blob)
        detections = self.ssd.forward() # [?, ?, num_detections, [?, class, confidence, x1, y1, x2, y2]]
        f = (detections[0, 0, :, 1] == self.person_class) & (detections[0, 0, :, 2] > self.confidence)
        detections = detections[0, 0, f, 2:7]
        detections = np.roll(detections, -1, axis=1)
        t_range = np.array([[0.0, w], [0.0, h]])

        if detections.size != 0:
            detections[:, :4] = self.pm.scale(detections[:, :4], self.s_range, t_range)

        return detections
