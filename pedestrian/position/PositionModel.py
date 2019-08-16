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

    def to_two_corners(self, position):
        """Convert a position to two corners format

        :param np.ndarray position: (batchSize, seqLength, target_dim(x1, y1, x2, y2))
        """
        pass

    def plot(self, frame, position, color: tuple):
        """ Plot (inside) the position in a frame.

            :param np.ndarray frame: The frame in BGR
            :param np.ndarray position: The objects's position representation
            :param tuple color: The BGR code of position's color.
        """
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

    def scale(self, position, s_range, t_range):
        """ Scale a position from source range to target range

        :param position: [..., coordinates]
        :param s_range: The source range [[xMin, xMax], [yMin, yMax]]
        :param t_range: The target range [[xMin, xMax], [yMin, yMax]]
        :return:
        """
        shape = position.shape
        o_diff = np.abs(s_range[:, :1] - s_range[:, 1:])  # [[xDiff], [yDiff]]
        t_diff = np.abs(t_range[:, :1] - t_range[:, 1:])  # [[xDiff], [yDiff]]
        position = position.reshape((-1, 2)).T
        position = t_diff * (position - s_range[:, :1]) / o_diff + t_range[:, :1]
        position = position.T.reshape(shape)

        return position
