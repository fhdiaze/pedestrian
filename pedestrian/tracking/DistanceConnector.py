import numpy as np
from pedestrian.position.CentroidPM import CentroidPM
from pedestrian.tracking.Connector import Connector
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment as linear_assignment


class DistanceConnector(Connector):

    __slots__ = ["cpm", "max_distance"]

    def __init__(self, max_distance: float = 50.0):
        self.cpm = CentroidPM(1.0, 1.0)
        self.max_distance = max_distance

    def connect(self, dets, tracks):
        dets = self.cpm.from_two_corners(dets[:, :-1])
        tracks = self.cpm.from_two_corners(tracks[:, :-1])
        D = dist.cdist(dets, tracks)
        matches = np.column_stack(linear_assignment(D))
        matches = np.column_stack([matches, D[matches[:, 0], matches[:, 1]]])

        return matches[matches[:, -1] < self.max_distance, :]
