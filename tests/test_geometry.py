import unittest

import numpy as np

from tennisvision.geometry import smooth_court_keypoints


class SmoothCourtKeypointsTests(unittest.TestCase):
    def test_rejects_impulse_without_freezing_motion(self):
        n_frames = 9
        x = np.arange(n_frames, dtype=np.float64)
        points = np.zeros((n_frames, 1, 2), dtype=np.float64)
        points[:, 0, 0] = 100.0 + 2.0 * x
        points[:, 0, 1] = 200.0 + x
        points[4, 0] += np.array([80.0, 60.0])

        smoothed = smooth_court_keypoints(
            points, np.ones(n_frames, dtype=bool), window=5)

        np.testing.assert_allclose(smoothed[4, 0], [110.0, 205.0])
        self.assertGreater(smoothed[-1, 0, 0], smoothed[0, 0, 0])

    def test_does_not_cross_invisible_camera_cut(self):
        points = np.full((7, 1, 2), np.nan)
        points[:3, 0] = [10.0, 20.0]
        points[4:, 0] = [1000.0, 800.0]
        visible = np.array([True, True, True, False, True, True, True])

        smoothed = smooth_court_keypoints(points, visible, window=5)

        self.assertTrue(np.isnan(smoothed[3]).all())
        np.testing.assert_allclose(smoothed[2, 0], [10.0, 20.0])
        np.testing.assert_allclose(smoothed[4, 0], [1000.0, 800.0])

    def test_rejects_invalid_windows(self):
        for window in (0, 2, 4):
            with self.subTest(window=window), self.assertRaises(ValueError):
                smooth_court_keypoints(
                    np.zeros((3, 1, 2)),
                    np.ones(3, dtype=bool),
                    window=window,
                )


if __name__ == "__main__":
    unittest.main()
