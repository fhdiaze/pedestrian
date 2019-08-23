import numpy as np
from pedestrian.position.CentroidPM import CentroidPM


class Track(object):
    UP = 1
    DOWN = -1
    STATIC = 0
    MAX_TRACKS = 150
    MIN_MOVE = 8
    COLORS = np.random.uniform(0, 255, size=(MAX_TRACKS, 3))

    __slots__ = ["positions", "counted", "cpm"]

    def __init__(self, position):
        self.positions = position
        self.counted = False
        self.cpm = CentroidPM(1.0, 1.0)

    def add(self, position):
        self.positions = np.vstack([self.positions, position])

    def direction(self):
        centroids = self.cpm.from_two_Corners(self.positions)
        direction = Track.STATIC

        if centroids.shape[0] > 1:
            delta = centroids[-1, 1] - np.mean(centroids[:-1, 1])
            if delta > Track.MIN_MOVE:
                direction = Track.DOWN
            elif delta < -Track.MIN_MOVE:
                direction = Track.UP

        return direction

    def intersect(self, line):
        r = False
        centroids = self.cpm.from_two_Corners(self.positions)
        pi = centroids[0, :]
        direction = self.direction()

        if (direction == Track.UP and line[0, 1] > centroids[-1, 1] and np.any(centroids[:, 1] > line[0, 1]))\
                or (direction == Track.DOWN and line[0, 1] < centroids[-1, 1] and np.any(centroids[:, 1] < line[0, 1])):
            r = True

        return r

    @classmethod
    def color(cls, idx: int):
        return cls.COLORS[idx % cls.MAX_TRACKS]