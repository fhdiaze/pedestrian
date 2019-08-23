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
        matched_indices = np.column_stack(linear_assignment(D))

        return matched_indices