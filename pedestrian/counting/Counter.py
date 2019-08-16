import numpy as np
from pedestrian.detection import Detector
from pedestrian.tracking import Tracker


class Counter(object):
    __slots__ = ["detector", "tracker", "det_period"]

    def __init__(self, detector: Detector, tracker: Tracker, det_period: int):
        self.detector = detector
        self.tracker = tracker
        self.det_period = det_period

    def count(self, frames):
        length = frames.shape[0]

        for t in range(length):
            dets = np.empty()

            if t % self.det_period == 0:
                dets = self.detector.detect(frames[t:t+1, ...])

            tracks = self.tracker.track(dets)
