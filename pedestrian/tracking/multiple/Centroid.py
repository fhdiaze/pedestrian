import dlib
import cv2
import numpy as np
from pedestrian.tracking.DistanceConnector import DistanceConnector
from pedestrian.tracking.multiple.Tracker import Tracker
from pedestrian.tracking.single.CorrelationTracker import CorrelationTracker


class Centroid(Tracker):

    def __init__(self, connector, detector, det_period: int = 5, max_unseen: int = 1):
        super().__init__(connector, detector, det_period, max_unseen)

    def update(self, frame, dets):
        if dets is not None:
            if dets.size != 0:
                tracks = np.zeros((len(self.trackers), 5))  # [x1, y1, x2, y2, idx]
                for i, (idx, t) in enumerate(self.trackers.items()):
                    (x1, y1, x2, y2) = t.track.positions[-1, :]
                    tracks[i, :] = [x1, y1, x2, y2, idx]

                matches = self.connector.connect(dets, tracks)

                for i in range(dets.shape[0]):
                    if i not in matches[:, 0]:
                        tracker = CorrelationTracker(self.next_id(), frame, dets[i, :-1])
                        self.trackers[tracker.idx] = tracker
                    else:
                        (det, idx, _) = matches[matches[:, 0] == i, :][0, :].astype("int")
                        idx = tracks[idx, -1].astype("int")
                        tracker = self.trackers.get(idx)
                        tracker.restart(frame, dets[det, :-1])

                for i in range(tracks.shape[0]):
                    if i not in matches[:, 1]:
                        idx = tracks[i, -1].astype("int")
                        self.trackers[idx].unseen += 1

            else:
                self.trackers.clear()
        else:
            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            for idx, tracker in self.trackers.items():
                tracker.update(frame)

        positions = np.zeros((0, 5))    # [x1, y1, x2, y2, idx]
        lost = []
        for idx, tracker in self.trackers.items():
            if tracker.unseen > self.max_unseen:
                lost.append(idx)
            else:
                (x1, y1, x2, y2) = tracker.last_position()
                positions = np.vstack([positions, [x1, y1, x2, y2, idx]])

        for idx in lost:
            del self.trackers[idx]

        return positions
