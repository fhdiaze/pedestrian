

class Tracker(object):

    __slots__ = ["frame_count", "idx", "trackers", "connector", "detector", "det_period", "max_unseen"]

    def __init__(self, connector, detector, det_period, max_unseen):
        self.frame_count = -1
        self.idx = -1
        self.trackers = dict()
        self.connector = connector
        self.detector = detector
        self.det_period = det_period
        self.max_unseen = max_unseen

    def next_id(self):
        self.idx += 1

        return self.idx

    def track(self, frame):
        """This method must be called once for each frame

        :param np.ndarray frame: a numpy array in the BGR format
        :return: a numpy array of tracks in the format [[x1, y1, x2, y2, id], [x1, y1, x2, y2, id], ...]
        :rtype: np.ndarray
        """
        self.frame_count += 1
        dets = None

        if self.frame_count % self.det_period == 0:
            dets = self.detector.detect(frame)

        tracks = self.update(frame, dets)

        return tracks

    def update(self, frame, detections):
        """This method must be called once for each frame even with empty or None detections.
        NOTE: None detections means that detection phase was not executed on this frame

        :param np.ndarray frame: a numpy array in the BGR format
        :param np.ndarray detections: a numpy array of detections in the format [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        :return: a numpy array of tracks in the format [[x1, y1, x2, y2, id], [x1, y1, x2, y2, id], ...]
        :rtype: np.ndarray
        """
        pass