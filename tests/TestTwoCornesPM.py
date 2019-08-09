import unittest
import numpy as np
from pedestrian.position.TwoCornersPM import TwoCornersPM


class TestTwoCornersPM(unittest.TestCase):
    def test_scale_single(self):
        pm = TwoCornersPM()
        frame_size = 416
        p = np.array([[0, 208, 208, 208]])
        s_range = np.array([[0.0, frame_size], [0.0, frame_size]])
        t_range = np.array([[0.0, 1.0], [0.0, 1.0]])

        gt = np.array([[0, 0.5, 0.5, 0.5]])
        pd = pm.scale(p, s_range, t_range)

        self.assertTrue((gt == pd).all())

    def test_scale_many(self):
        pm = TwoCornersPM()
        frame_size = 416
        p = np.array([[0, 0, 208, 208], [208, 208, 416, 416]])
        s_range = np.array([[0.0, frame_size], [0.0, frame_size]])
        t_range = np.array([[0.0, 1.0], [0.0, 1.0]])

        gt = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
        pd = pm.scale(p, s_range, t_range)

        self.assertTrue((gt == pd).all())

    def test_iou(self):
        pm = TwoCornersPM()
        frame_size = 416
        gtp = np.array([[0, 0, 208, 208], [208, 208, 416, 416]])
        prp = np.array([[0, 0, 208, 208], [208, 208, 416, 416]])

        iou = pm.iou(gtp, prp)

        self.assertTrue((iou == 1.0).all())
