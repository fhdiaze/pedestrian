import cv2
import numpy as np
from pedestrian.detection import Detector
from pedestrian.position.CentroidPM import CentroidPM
from pedestrian.tracking.multiple import Tracker
from pedestrian.tracking.core.Track import Track


class Counter(object):
    __slots__ = ["tracker", "line", "up", "down", "pm"]

    def __init__(self, tracker: Tracker, line=None):
        self.tracker = tracker
        self.line = line
        self.up = 0
        self.down = 0
        self.pm = CentroidPM(1.0, 1.0)

    def update(self, frame):
        """Updates the count status of the counter

        :param np.ndarray frame: a numpy array in the BGR format
        """
        tracks = self.tracker.track(frame)

        for track in tracks:
            (x1, y1, x2, y2, idx) = track.astype("int")
            self.pm.plot(frame, self.pm.from_two_corners(np.array([x1, y1, x2, y2])), Track.color(idx), idx)

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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    def count(self):
        """Updates the pedestrians counting
        """
        for (idx, tracker) in self.tracker.trackers.items():
            if not tracker.track.counted:
                direction = tracker.track.direction()

                if tracker.track.intersect(self.line) and direction != Track.STATIC:
                    tracker.track.counted = True
                    if direction == Track.UP:
                        self.up += 1
                    elif direction == Track.DOWN:
                        self.down += 1
