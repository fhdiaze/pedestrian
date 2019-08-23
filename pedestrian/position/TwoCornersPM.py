import cv2
import numpy as np
from pedestrian.position.PositionModel import PositionModel


class TwoCornersPM(PositionModel):

    def __init__(self):
        super(TwoCornersPM, self).__init__(4)

    def from_two_corners(self, position):
        """Transform positions from two corners format to another format

        :param position: (batchSize, seqLength, [x1, y1, x2, y2])
        :return: (batchSize, seqLength, [x1, y1, x2, y2])
        """
        return position

    def to_two_corners(self, position):
        """Convert a position to two corners format

        :param np.ndarray position: (batchSize, seqLength, [x1, y1, x2, y2])
        """
        return position

    def plot(self, frame, position, color: tuple, idx: str = None, thickness: int = 2):
        """Plots (in side) the position in a frame.

        :param np.ndarray frame: The frame in BGR
        :param np.ndarray position: The objects's corners coordinates [x1, y1, x2, y2]
        :param tuple color: The BGR code of position's color.
        :param str idx: the position's id
        :param int thickness:
        """
        (x1, y1, x2, y2) = position
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    def iou(self, gt_position, pred_position):
        """Calculate the intersection over union of positions

        :param gt_position: the ground truth position. (batchSize, seqLength, target_dim(x1, y1, x2, y2))
        :param pred_position: the predicted position
        :return:
        """
        gtp_shape = gt_position.shape

        gt_position = gt_position.reshape((-1, self.target_dim))
        pred_position = pred_position.reshape((-1, self.target_dim))

        left = np.max([pred_position[:, 0], gt_position[:, 0]], axis=0)
        top = np.max([pred_position[:, 1], gt_position[:, 1]], axis=0)
        right = np.min([pred_position[:, 2], gt_position[:, 2]], axis=0)
        bottom = np.min([pred_position[:, 3], gt_position[:, 3]], axis=0)

        intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
        label_area = np.abs(gt_position[..., 2] - gt_position[..., 0]) * np.abs(gt_position[..., 3] - gt_position[..., 1])
        predict_area = np.abs(pred_position[..., 2] - pred_position[..., 0]) * np.abs(
            pred_position[..., 3] - pred_position[..., 1])
        union = label_area + predict_area - intersect

        iou = intersect / union

        return iou.reshape(gtp_shape[:-1]+(1,))
