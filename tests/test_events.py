import unittest

import numpy as np

from tennisvision.analytics import compute_stats
from tennisvision.events import detect_hits


class EventDetectionTests(unittest.TestCase):
    def test_does_not_bridge_long_camera_cut(self):
        ball_court = np.full((80, 2), np.nan)
        ball_px = np.full((80, 2), np.nan)
        ball_court[5:20, 1] = np.linspace(14.0, 8.0, 15)
        ball_court[5:20, 0] = 5.0
        ball_px[5:20] = np.column_stack([
            np.full(15, 100.0), np.linspace(400.0, 100.0, 15)])
        ball_court[60:75, 1] = np.linspace(8.0, 14.0, 15)
        ball_court[60:75, 0] = 5.0
        ball_px[60:75] = np.column_stack([
            np.full(15, 100.0), np.linspace(100.0, 400.0, 15)])

        hits = detect_hits(ball_court, 25.0, ball_px, [{}] * 80)

        self.assertEqual(hits, [])

    def test_stats_counts_hit_when_only_striker_is_tracked(self):
        ball = np.full((20, 2), np.nan)
        ball[5] = [5.0, 22.0]
        players = [{} for _ in range(20)]
        players[5] = {1: np.array([5.0, 23.0])}

        stats = compute_stats([5], ball, players, 25.0)

        self.assertEqual(stats.shot_count(1), 1)

    def test_explicit_contact_player_overrides_airborne_projection(self):
        ball = np.full((20, 2), np.nan)
        ball[5] = [5.0, 10.0]
        players = [{} for _ in range(20)]
        players[5] = {
            1: np.array([5.0, 23.0]),
            2: np.array([5.0, 1.0]),
        }

        stats = compute_stats(
            [5], ball, players, 25.0, hit_players={5: 1})

        self.assertEqual(stats.shot_count(1), 1)
        self.assertEqual(stats.shot_count(2), 0)

    def test_counts_terminal_contact_without_outgoing_branch(self):
        ball_court = np.column_stack([
            np.full(20, 5.0),
            np.linspace(8.0, 23.0, 20),
        ])
        ball_px = np.column_stack([
            np.full(20, 100.0),
            np.linspace(200.0, 500.0, 20),
        ])
        boxes = [{} for _ in range(20)]
        boxes[-1] = {1: np.array([80.0, 450.0, 120.0, 520.0])}

        events = detect_hits(
            ball_court, 25.0, ball_px, boxes, return_players=True)

        self.assertEqual(events[-1][1], 1)
        self.assertGreaterEqual(events[-1][0], 18)


if __name__ == "__main__":
    unittest.main()
