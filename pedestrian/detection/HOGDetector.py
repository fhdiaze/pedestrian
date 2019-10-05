import cv2
import numpy as np
from pedestrian.detection.Detector import Detector
from pedestrian.position.TwoCornersPM import TwoCornersPM


class HOGDetector(Detector):
    __slots__ = ["in_size", "stride", "padding", "mean_shift", "scale", "confidence", "s_range", "pm", "hog"]

    def __init__(self, stride=(2, 2), padding=(16, 16), mean_shift=False, scale=1.01, confidence: float = 1.0):
        self.in_size = (600, 600)   # w, h
        self.stride = stride
        self.padding = padding
        self.mean_shift = mean_shift
        self.scale = scale
        self.confidence = confidence
        self.s_range = np.array([[0.0, self.in_size[0]], [0.0, self.in_size[1]]])
        self.pm = TwoCornersPM()
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        (h, w) = frame.shape[:2]
        frame = cv2.resize(frame, self.in_size)
        # [?, ?, num_detections, [?, class, confidence, x1, y1, x2, y2]]
        (dets, weights) = self.hog.detectMultiScale(frame, winStride=self.stride, padding=self.padding, scale=self.scale, useMeanshiftGrouping=self.mean_shift)
        dets = np.hstack([dets, weights])

        if dets.size != 0:
            dets[:, 2] += dets[:, 0]
            dets[:, 3] += dets[:, 1]
            f = (dets[:, -1] > self.confidence)
            dets = dets[f, :]
            t_range = np.array([[0.0, w], [0.0, h]])
            dets[:, :4] = self.pm.scale(dets[:, :4], self.s_range, t_range)

        return dets
