import numpy as np
from pedestrian.position.CentroidPM import CentroidPM
from pedestrian.tracking.Connector import Connector
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment as linear_assignment


class DistanceConnector(Connector):

    def __init__(self):
        self.cpm = CentroidPM(1.0, 1.0)

    def connect(self, dets, trks):
        dets = self.cpm.from_two_Corners(dets[:, :-1])
        trks = self.cpm.from_two_Corners(trks[:, :-1])
        D = dist.cdist(dets, trks)
        matches = np.column_stack(linear_assignment(D))
        np.column_stack([matches, D[matches[:, 0], matches[:, 0]]])

        return matches[matches[:, -1] < 10, :]
