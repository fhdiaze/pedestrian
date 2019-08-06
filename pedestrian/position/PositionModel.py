import numpy as np


class PositionModel(object):

    def __init__(self, target_dim):
        self.target_dim = target_dim

    def get_target_dim(self):
        return self.target_dim

    def from_two_corners(self, position):
        """Transform positions from two corners format to another format

        :param position: (batchSize, seqLength, target_dim(x1, y1, x2, y2))
        :return:
        """
        pass

    # position.shape = (batchSize, seqLength, target_dim(x1, y1, x2, y2))
    def to_two_corners(self, position):
        pass

    """
    Plot (inside) the position in a frame. 
    @type  frame:    PIL.Image
    @param frame:    The frame
    @type  position: object
    @param position: The objects's position representation
    @type  outline:  string
    @param outline:  The name of the position color.
    """

    def plot(self, frame, position, outline):
        pass

    # theta.shape = (batchSize, 3, 3)
    def transform(self, theta, position):
        position_shape = position.shape
        theta = theta.reshape((-1, 3, 3))
        samples = theta.shape[0]
        position = position.reshape((samples, -1, 2)).transpose(0, 2, 1)

        # Reshaping the positions
        position = np.concatenate((position, np.ones((samples, 1, position.shape[2]))), axis=1)

        # Applying the transformation
        position = np.matmul(theta, position)[:, :2, :]

        # Reshaping the result
        position = position.transpose(0, 2, 1)
        position = position.reshape(position_shape)

        return position

    # position = [..., coordinates]
    # sRange = [[xMin, xMax], [yMin, yMax]]
    # tRange = [[xMin, xMax], [yMin, yMax]]
    def scale(self, position, s_range, t_range):
        """
        Scale a position from source range to target range
        :param position:
        :param s_range: The source range
        :param t_range: The target range
        :return:
        """
        shape = position.shape
        o_diff = np.abs(s_range[:, :1] - s_range[:, 1:])  # [[xDiff], [yDiff]]
        t_diff = np.abs(t_range[:, :1] - t_range[:, 1:])  # [[xDiff], [yDiff]]
        position = position.reshape((-1, 2)).T
        position = t_diff * (position - s_range[:, :1]) / o_diff + t_range[:, :1]
        position = position.T.reshape(shape)

        return position
