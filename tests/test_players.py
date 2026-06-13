import unittest

import numpy as np

from tennisvision.detect.players import PlayerTracker
from tennisvision.geometry import COURT_LENGTH


class IdentityCourt:
    def to_court(self, points):
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def box_at(x, y, confidence=0.9):
    return np.array([x - 0.5, y - 2.0, x + 0.5, y, confidence])


class PlayerSelectionTests(unittest.TestCase):
    def test_accepts_track_id_changes_in_each_half(self):
        detections = [
            {10: box_at(5, 1), 20: box_at(5, 23)},
            {10: box_at(6, 1), 21: box_at(6, 22)},
            {11: box_at(7, 2), 21: box_at(7, 22)},
        ]

        selected = PlayerTracker.select_players(
            detections, [IdentityCourt()] * len(detections))

        self.assertTrue(all(set(frame) == {1, 2} for frame in selected))
        np.testing.assert_allclose(selected[1][1], detections[1][21])
        np.testing.assert_allclose(selected[2][2], detections[2][11])

    def test_ignores_official_near_net_when_player_is_missing(self):
        detections = [
            {10: box_at(5, 1), 20: box_at(5, 23)},
            {10: box_at(5, 1), 90: box_at(11.8, 13.5)},
            {10: box_at(5, 1), 20: box_at(7, 23)},
        ]

        selected = PlayerTracker.select_players(
            detections, [IdentityCourt()] * len(detections), max_gap=2)

        self.assertEqual(set(selected[1]), {1, 2})
        self.assertTrue(np.isnan(selected[1][1][4]))
        self.assertAlmostEqual(selected[1][1][3], 23.0)

    def test_keeps_spatially_continuous_player_away_from_baseline(self):
        detections = [
            {10: box_at(5, 1), 20: box_at(5, 18)},
            {10: box_at(5, 1), 21: box_at(5.5, 16)},
            {10: box_at(5, 1), 22: box_at(6, 14.5)},
        ]

        selected = PlayerTracker.select_players(
            detections, [IdentityCourt()] * len(detections))

        self.assertIn(1, selected[2])
        np.testing.assert_allclose(selected[2][1], detections[2][22])

    def test_does_not_interpolate_long_gaps(self):
        detections = [
            {10: box_at(5, 1), 20: box_at(5, COURT_LENGTH)},
            {10: box_at(5, 1)},
            {10: box_at(5, 1)},
            {10: box_at(5, 1), 20: box_at(8, COURT_LENGTH)},
        ]

        selected = PlayerTracker.select_players(
            detections, [IdentityCourt()] * len(detections), max_gap=1)

        self.assertNotIn(1, selected[1])
        self.assertNotIn(1, selected[2])


if __name__ == "__main__":
    unittest.main()
