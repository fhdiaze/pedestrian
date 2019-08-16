import unittest
import numpy as np
from pedestrian.position.TwoCornersPM import TwoCornersPM


class TestTwoCornersPM(unittest.TestCase):
    def test_scale_single(self):
        # Assume
        pm = TwoCornersPM()
        frame_size = 416
        p = np.array([[0, 208, 208, 208]])
        s_range = np.array([[0.0, frame_size], [0.0, frame_size]])
        t_range = np.array([[0.0, 1.0], [0.0, 1.0]])
        gt = np.array([[0, 0.5, 0.5, 0.5]])

        # Action
        pd = pm.scale(p, s_range, t_range)

        # Assert
        self.assertTrue((gt == pd).all())

    def test_scale_many(self):
        # Assume
        pm = TwoCornersPM()
        frame_size = 416
        p = np.array([[0, 0, 208, 208], [208, 208, 416, 416]])
        s_range = np.array([[0.0, frame_size], [0.0, frame_size]])
        t_range = np.array([[0.0, 1.0], [0.0, 1.0]])
        gt = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])

        # Action
        pd = pm.scale(p, s_range, t_range)

        # Assert
        self.assertTrue((gt == pd).all())

    def test_scale_many2(self):
        # Assume
        pm = TwoCornersPM()
        frame_size = 416
        p = np.array([[0, 0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
        s_range = np.array([[0.0, 1.0], [0.0, 1.0]])
        t_range = np.array([[0.0, frame_size], [0.0, frame_size]])
        gt = np.array([[0.0, 0.0, 208, 208], [208, 208, 416, 416]])

        # Action
        pd = pm.scale(p, s_range, t_range)

        # Assert
        self.assertTrue((gt == pd).all())

    def test_iou(self):
        # Assume
        pm = TwoCornersPM()
        frame_size = 416
        gtp = np.array([[0, 0, 208, 208], [208, 208, 416, 416]])
        prp = np.array([[0, 0, 208, 208], [208, 208, 416, 416]])

        # Action
        iou = pm.iou(gtp, prp)

        # Assert
        self.assertTrue((iou == 1.0).all())
