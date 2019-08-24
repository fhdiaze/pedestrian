import numpy as np


class Detector(object):

    __slots__ = ["confidence"]

    def __init__(self, confidence: float):
        self.confidence = confidence

    def detect(self, frame):
        """Finds pedestrians in a frame

        :param np.ndarray frame: BGR format
        :return: (samples, [x1, y1, x2, y2, score])
        :rtype: np.ndarray
        """
        pass
