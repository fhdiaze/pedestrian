import cv2
import numpy as np
from pedestrian.detection import Detector
from pedestrian.tracking import Tracker
from pedestrian.tracking.Track import Track


class Counter(object):
    __slots__ = ["frame_count", "detector", "det_period", "tracker", "tracks", "line", "up", "down"]

    def __init__(self, detector: Detector, det_period: int, tracker: Tracker):
        self.frame_count = 0
        self.detector = detector
        self.tracker = tracker
        self.det_period = det_period
        self.tracks = dict()
        self.line = np.array([[0, 150], [150, 150]])
        self.up = 0
        self.down = 0

    def update(self, frame):
        """Updates the count status of the counter

        :param np.ndarray frame: a numpy array in the BGR format
        """

        boxes = np.empty((0, 5))

        if self.frame_count % self.det_period == 0:
            boxes = self.detector.detect(frame)

        tracks = self.tracker.track(frame, boxes.astype("int"))

        for track in tracks:
            (x1, y1, x2, y2, idx) = track.astype("int")
            track = self.tracks.get(idx, Track(idx, np.empty((0, 4))))
            track.add(np.array([x1, y1, x2, y2]))
            self.tracks[idx] = track

        self.count()

    def count(self):

        for (oid, track) in self.tracks.items():
            if not track.counted:
                direction = track.direction()

                if track.intersect(self.line):
                    track.counted = True
                    if direction == Track.UP:
                        self.up += 1
                    elif direction == Track.DOWN:
                        self.down += 1
