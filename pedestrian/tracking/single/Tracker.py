from pedestrian.tracking.core.Track import Track


class Tracker(object):

    __slots__ = ["idx", "track", "unseen"]

    def __init__(self, idx: int, position):
        self.idx = idx
        self.track = Track(position)
        self.unseen = 0

    def last_position(self):
        return self.track.positions[-1, :]

    def update(self, frame):
        pass

    def restart(self, det):
        self.unseen = 0
        self.track.add(det)
