import cv2
import dlib
import numpy as np
from pedestrian.tracking.single.Tracker import Tracker


class CorrelationTracker(Tracker):

    __slots__ = ["motion"]

    def __init__(self, idx: int, frame, position):
        super().__init__(idx, frame, position)
        self.motion = dlib.correlation_tracker()
        (x1, y1, x2, y2) = position.astype("int")
        rect = dlib.rectangle(x1, y1, x2, y2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.motion.start_track(rgb, rect)

    def update(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.motion.update(rgb)
        pos = self.motion.get_position()

        # unpack the position object
        x1 = int(pos.left())
        y1 = int(pos.top())
        x2 = int(pos.right())
        y2 = int(pos.bottom())

        npos = np.array([x1, y1, x2, y2])
        self.track.add(npos)

        return npos

    def restart(self, frame, pos):
        super().restart(frame, pos)
        (x1, y1, x2, y2) = pos.astype("int")
        rect = dlib.rectangle(x1, y1, x2, y2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.motion.start_track(rgb, rect)
