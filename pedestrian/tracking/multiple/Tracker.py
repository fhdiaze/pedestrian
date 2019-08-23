

class Tracker(object):

    __slots__ = ["frame_count", "idx", "trackers", "connector", "max_unseen"]

    def __init__(self, connector, max_unseen: int = 1):
        self.frame_count = 0
        self.idx = 0
        self.trackers = dict()
        self.connector = connector
        self.max_unseen = max_unseen

    def next_id(self):
        self.idx += 1

        return self.idx

    def track(self, frame, detections):
        """This method must be called once for each frame even with empty detections.
        NOTE: The number of objects returned may differ from the number of detections provided.

        :param np.ndarray frame: a numpy array in the BGR format
        :param np.ndarray detections: a numpy array of detections in the format [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        :return: a numpy array of detections in the format [[x1, y1, x2, y2, id], [x1, y1, x2, y2, id], ...]
        :rtype: np.ndarray
        """
        pass