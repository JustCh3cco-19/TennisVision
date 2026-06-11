"""Match statistics computed directly in metric court space.

Because all positions are already in meters (via the homography), speeds
and distances need no pixel-to-meter conversion factor.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ShotStat:
    frame: int
    player: int          # 1 or 2
    ball_speed_kmh: float
    opponent_speed_kmh: float


@dataclass
class MatchStats:
    shots: list = field(default_factory=list)
    fps: float = 25.0

    def shot_count(self, player: int) -> int:
        return sum(1 for s in self.shots if s.player == player)

    def avg_shot_speed(self, player: int) -> float:
        v = [s.ball_speed_kmh for s in self.shots if s.player == player]
        return float(np.mean(v)) if v else 0.0

    def last_shot_speed(self, player: int, frame: int) -> float:
        v = [s.ball_speed_kmh for s in self.shots
             if s.player == player and s.frame <= frame]
        return v[-1] if v else 0.0


def compute_stats(hits: list, ball_court: np.ndarray,
                  players_court: list, fps: float) -> MatchStats:
    """hits: shot frame indices; ball_court: (N,2) meters;
    players_court: per-frame dict {1: (x,y), 2: (x,y)} in meters."""
    stats = MatchStats(fps=fps)
    for a, b in zip(hits[:-1], hits[1:]):
        dt = (b - a) / fps
        if dt <= 0 or not np.isfinite(ball_court[[a, b]]).all():
            continue
        ball_dist = float(np.linalg.norm(ball_court[b] - ball_court[a]))
        ball_speed = ball_dist / dt * 3.6

        # striker = player closest to the ball at the moment of the hit
        pos = players_court[a]
        if len(pos) < 2:
            continue
        striker = min(pos, key=lambda p: np.linalg.norm(
            np.asarray(pos[p]) - ball_court[a]))
        opponent = 2 if striker == 1 else 1

        opp_a = players_court[a].get(opponent)
        opp_b = players_court[b].get(opponent)
        opp_speed = 0.0
        if opp_a is not None and opp_b is not None:
            opp_speed = float(np.linalg.norm(
                np.asarray(opp_b) - np.asarray(opp_a))) / dt * 3.6

        stats.shots.append(ShotStat(frame=a, player=striker,
                                    ball_speed_kmh=ball_speed,
                                    opponent_speed_kmh=opp_speed))
    return stats
