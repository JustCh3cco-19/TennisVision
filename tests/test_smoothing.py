import unittest

import numpy as np

from tennisvision.geometry import CourtReference
from tennisvision.smoothing import (
    BallParabolicSmoother,
    filter_ball_detections_to_court,
    filter_static_ball_detections,
)


class BallSmoothingTests(unittest.TestCase):
    def test_impossible_early_detection_does_not_poison_later_track(self):
        points = np.full((20, 2), np.nan)
        points[0] = [500.0, 500.0]
        frames = np.arange(3, 13)
        points[frames, 0] = 100.0 + 10.0 * frames
        points[frames, 1] = 200.0 + 2.0 * (frames - 7.0) ** 2

        smoothed = BallParabolicSmoother().smooth(points)

        self.assertTrue(np.isfinite(smoothed[3:13]).all())
        self.assertTrue(np.isnan(smoothed[0]).all())

    def test_stationary_false_positive_is_not_fitted(self):
        points = np.full((20, 2), np.nan)
        points[2:15] = [80.0, 120.0]

        smoothed = BallParabolicSmoother().smooth(points)

        self.assertTrue(np.isnan(smoothed).all())

    def test_repeated_static_hotspot_is_removed_before_fitting(self):
        points = np.full((30, 3), np.nan)
        points[[2, 5, 8, 11, 14], :2] = [420.0, 640.0]
        points[[2, 5, 8, 11, 14], 2] = 0.4
        points[20:24, :2] = np.array([
            [100.0, 200.0],
            [110.0, 190.0],
            [120.0, 184.0],
            [130.0, 182.0],
        ])
        points[20:24, 2] = 0.8

        filtered = filter_static_ball_detections(points)

        self.assertTrue(np.isnan(filtered[[2, 5, 8, 11, 14]]).all())
        self.assertTrue(np.isfinite(filtered[20:24]).all())

    def test_court_view_filter_keeps_high_ball_and_rejects_far_overlay(self):
        keypoints = np.array([
            [100.0, 100.0],
            [300.0, 100.0],
            [100.0, 300.0],
            [300.0, 300.0],
        ])
        court = CourtReference.from_keypoints(keypoints)
        detections = np.array([
            [200.0, 30.0, 0.8],
            [-30.0, 200.0, 0.7],
        ])

        filtered = filter_ball_detections_to_court(
            detections, [court, court])

        np.testing.assert_allclose(filtered[0], detections[0])
        self.assertTrue(np.isnan(filtered[1]).all())


if __name__ == "__main__":
    unittest.main()
