

class Tracker(object):

    def track(self, dets):
        """Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.

        :param dets: a numpy array of detections in the format [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        :return: a numpy array of detections in the format [[x1, y1, x2, y2, id], [x1, y1, x2, y2,id], ...]
        :rtype: ndarray
        """
        pass