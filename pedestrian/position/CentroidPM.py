import numpy as np
from pedestrian.position.PositionModel import PositionModel


class CentroidPM(PositionModel):

    def __init__(self, height, width):
        super(CentroidPM, self).__init__(2)
        self.height = height
        self.width = width

    # position.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def from_two_Corners(self, position):
        shape = position.shape
        position = position.reshape((-1, 2, 2))
        centroid = np.sum(position, axis=1) / 2.0
        new_position = centroid.reshape(shape[:-1] + (2,))

        return new_position

    # position.shape = (batchSize, seqLength, targetDim(xC, yC))
    def to_two_corners(self, position):
        shape = position.shape
        position = position.reshape((-1, 2))
        x_center = position[:, 0]
        y_center = position[:, 1]

        new_position = np.zeros((position.shape[0], 4))
        new_position[:, 0] = x_center - self.width / 2.0
        new_position[:, 1] = y_center - self.height / 2.0
        new_position[:, 2] = x_center + self.width / 2.0
        new_position[:, 3] = y_center + self.height / 2.0
        new_position = new_position.reshape(shape[:-1] + (4,))

        return new_position

    """
    Plot (inside) the position in a frame. 
    @type  frame:    PIL.Image
    @param frame:    The frame
    @type  position: [integer, integer]
    @param position: The objects's centroid coordinates (x, y)
    @type  outline:  string
    @param outline:  The name of the position color.
    """

    def plot(self, frame, position, outline):
        """Plot (in side) the position in a frame.

        :param np.ndarray frame: The frame in BGR
        :param np.ndarray position: The objects's corners coordinates [x1, y1, x2, y2]
        :param tuple color: The BGR code of position's color.
        :param int thickness:
        """
        draw = ImageDraw.Draw(frame)
        w, h = frame.size
        x = position[0]
        y = position[1]
        draw.line([x, 0, x, h], fill=outline)
        draw.line([0, y, w, y], fill=outline)


        (x1, y1, x2, y2) = position
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)