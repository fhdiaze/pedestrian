

class Tracker(object):

    def track(self, frame, detections):
        """This method must be called once for each frame even with empty detections.
        NOTE: The number of objects returned may differ from the number of detections provided.

        :param np.ndarray frame: a numpy array in the BGR format
        :param np.ndarray detections: a numpy array of detections in the format [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        :return: a numpy array of detections in the format [[x1, y1, x2, y2, id], [x1, y1, x2, y2, id], ...]
        :rtype: np.ndarray
        """
        pass