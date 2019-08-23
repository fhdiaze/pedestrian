import cv2
import dlib
import numpy as np
from pedestrian.tracking.single.Tracker import Tracker


class CorrelationTracker(Tracker):

    __slots__ = ["motion"]

    def __init__(self, idx: int, position):
        super(Tracker, self).__init__(idx, position)
        pass

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
        super(Tracker, self).__init__(pos)
        (x1, y1, x2, y2) = pos.astype("int")
        rect = dlib.rectangle(x1, y1, x2, y2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.motion.start_track(rgb, rect)
