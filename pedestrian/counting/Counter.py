import cv2
import numpy as np
from pedestrian.detection import Detector
from pedestrian.position.CentroidPM import CentroidPM
from pedestrian.tracking.multiple import Tracker
from pedestrian.tracking.core.Track import Track


class Counter(object):
    __slots__ = ["frame_count", "detector", "det_period", "tracker", "tracks", "line", "up", "down", "pm"]

    def __init__(self, detector: Detector, det_period: int, tracker: Tracker, line=None):
        self.frame_count = 0
        self.detector = detector
        self.tracker = tracker
        self.det_period = det_period
        self.tracks = dict()
        self.line = line
        self.up = 0
        self.down = 0
        self.pm = CentroidPM(1.0, 1.0)

    def update(self, frame):
        """Updates the count status of the counter

        :param np.ndarray frame: a numpy array in the BGR format
        """

        dets = np.empty((0, 5))

        if self.frame_count % self.det_period == 0:
            dets = self.detector.detect(frame)

        tracks = self.tracker.track(frame, dets)

        for track in tracks:
            (x1, y1, x2, y2, idx) = track.astype("int")
            track = self.tracks.get(idx, Track(idx, np.empty((0, 4))))
            track.add(np.array([x1, y1, x2, y2]))
            self.tracks[idx] = track
            self.pm.plot(frame, self.pm.from_two_Corners(np.array([x1, y1, x2, y2])), Track.color(idx), idx)

        self.count()

        h, w, _ = frame.shape
        if self.line is not None:
            cv2.line(frame, tuple(self.line[0]), tuple(self.line[1]), (0, 255, 255), 2)

        info = [
            ("Up", self.up),
            ("Down", self.down),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, h - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def count(self):
        """Updates the pedestrians counting
        """
        for (oid, track) in self.tracks.items():
            if not track.counted:
                direction = track.direction()

                if track.intersect(self.line) and direction != Track.STATIC:
                    track.counted = True
                    if direction == Track.UP:
                        self.up += 1
                    elif direction == Track.DOWN:
                        self.down += 1
