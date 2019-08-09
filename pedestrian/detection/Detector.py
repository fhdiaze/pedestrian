import numpy as np


class Detector(object):

    def detect(self, frame):
        """Finds pedestrians in a frame

        :param np.ndarray frame: (h, w, 3)
        :return: (samples, [x1, y1, x2, y2, score])
        :rtype: np.ndarray
        """
        pass
