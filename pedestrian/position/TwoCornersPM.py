import numpy as np
from PIL import ImageDraw
from pedestrian.position.PositionModel import PositionModel


class TwoCornersPM(PositionModel):

    def __init__(self):
        super(TwoCornersPM, self).__init__(4)

    # position.shape = (batchSize, seqLength, target_dim(x1, y1, x2, y2))
    def from_two_corners(self, position):
        return position

    # position.shape = (batchSize, seqLength, target_dim(x1, y1, x2, y2))
    def to_two_corners(self, position):
        return position

    def plot(self, frame, position, outline: str):
        """ Plot (in side) the position in a frame.

        :param PIL.Image frame: The frame
        :param [integer, integer, integer, integer] position: The objects's corners coordinates [x1, y1, x2, y2]
        :param str outline:  The name of the position color.
        """
        draw = ImageDraw.Draw(frame)
        draw.rectangle(position, outline=outline)

    # gtPosition.shape = (batchSize, seqLength, target_dim(x1, y1, x2, y2))
    def iou(self, gt_position, pred_position):
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
