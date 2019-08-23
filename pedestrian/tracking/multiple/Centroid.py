import dlib
import cv2
import numpy as np
from pedestrian.tracking.DistanceConnector import DistanceConnector
from pedestrian.tracking.multiple.Tracker import Tracker
from pedestrian.tracking.single.CorrelationTracker import CorrelationTracker


class Centroid(Tracker):

    def __init__(self, connector):
        super(Tracker, self).__init__(connector)

    def track(self, frame, dets):
        if dets.size != 0:
            trks = np.zeros((len(self.trackers), 5))  # [x1, y1, x2, y2, idx]
            for i, (idx, t) in enumerate(self.trackers.items()):
                (x1, y1, x2, y2) = t.track.positions[-1, :]
                trks[i, :] = [x1, y1, x2, y2, idx]

            matches = DistanceConnector(10.0).connect(dets, trks[:, -1])

            for i in np.arange(0, dets.shape[0]):
                if i not in matches[:, 0]:
                    pos = dets[i, :]
                    (x1, y1, x2, y2) = pos.astype("int")
                    tracker = CorrelationTracker(self.next_id(), pos)

                    self.trackers[tracker.idx] = tracker
                else:
                    (det, idx, _) = matches[matches[:, 0] == i, :]
                    tracker = self.trackers.get(idx)
                    tracker.restart(frame, np.array([x1, y1, x2, y2]))

            for idx, tracker in self.trackers.items():
                if idx not in matches[:, 0]:
                    tracker.unseen += 1

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for idx, tracker in self.trackers.items():
                tracker.update(frame)

        positions = np.zeros((len(self.trackers), 5)) # [x1, y1, x2, y2, idx]
        for idx, tracker in self.trackers.items():
            if tracker.lost > self.max_unseen:
                del self.trackers[idx]
            positions = np.vstack([positions, [x1, y1, x2, y2, idx]])

        return positions
