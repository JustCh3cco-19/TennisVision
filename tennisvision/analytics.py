"""Match statistics computed directly in metric court space.

Because all positions are already in meters (via the homography), speeds
and distances need no pixel-to-meter conversion factor.
"""

from dataclasses import dataclass, field

import numpy as np

from .geometry import COURT_LENGTH, NET_Y, SERVICE_FROM_NET


@dataclass
class ShotStat:
    """Statistics of a single detected shot.

    Attributes:
        frame: Frame index where the shot is struck.
        player: Striking player, 1 (bottom half) or 2 (top half).
        ball_speed_kmh: Average ball speed until the next hit, km/h.
        opponent_speed_kmh: Opponent movement speed during the shot, km/h.
        shot_type: One of "serve", "volley", "groundstroke", classified
            geometrically from the striker's court position at the hit.
    """

    frame: int
    player: int
    ball_speed_kmh: float
    opponent_speed_kmh: float
    shot_type: str = "groundstroke"


def classify_shot(striker_pos, rally_start: bool) -> str:
    """Classifies a shot from the striker's metric court position.

    Geometric rules only (no appearance or pose information):
    a rally-opening hit struck from at or behind the baseline is a serve;
    a hit struck between the net and the service line is a volley;
    everything else is a groundstroke.

    Args:
        striker_pos: (x, y) striker position in court meters at the hit.
        rally_start: True if this hit opens a rally (first hit, or first
            after a long pause in the ball track).

    Returns:
        One of "serve", "volley", "groundstroke".
    """
    dist_from_net = abs(float(striker_pos[1]) - NET_Y)
    near_baseline = dist_from_net >= COURT_LENGTH / 2 - 1.0
    if rally_start and near_baseline:
        return "serve"
    if dist_from_net < SERVICE_FROM_NET:
        return "volley"
    return "groundstroke"


@dataclass
class MatchStats:
    """Aggregated match statistics.

    Attributes:
        shots: List of ShotStat, one per detected shot.
        fps: Video frame rate, used for time conversions.
        player_speeds: (N, 2) per-frame movement speed in km/h for
            players 1 and 2, or None if not computed.
    """

    shots: list = field(default_factory=list)
    fps: float = 25.0
    player_speeds: np.ndarray = None

    def shot_count(self, player: int, frame: int = None) -> int:
        """Counts the shots of a player.

        Args:
            player: Player id, 1 or 2.
            frame: If given, only count shots struck up to this frame.

        Returns:
            Number of shots.
        """
        return sum(1 for s in self.shots if s.player == player
                   and (frame is None or s.frame <= frame))

    def avg_shot_speed(self, player: int, frame: int = None) -> float:
        """Computes the mean ball speed of a player's shots.

        Args:
            player: Player id, 1 or 2.
            frame: If given, only consider shots struck up to this frame.

        Returns:
            Mean shot speed in km/h, 0.0 if the player has no shots.
        """
        v = [s.ball_speed_kmh for s in self.shots if s.player == player
             and s.ball_speed_kmh > 0
             and (frame is None or s.frame <= frame)]
        return float(np.mean(v)) if v else 0.0

    def last_shot_speed(self, player: int, frame: int) -> float:
        """Returns the speed of the player's most recent shot.

        Args:
            player: Player id, 1 or 2.
            frame: Current frame; shots after it are ignored.

        Returns:
            Ball speed of the last shot in km/h, 0.0 if none yet.
        """
        v = [s.ball_speed_kmh for s in self.shots
             if s.player == player and s.frame <= frame]
        return v[-1] if v else 0.0

    def last_shot_type(self, player: int, frame: int) -> str:
        """Returns the type of the player's most recent shot.

        Args:
            player: Player id, 1 or 2.
            frame: Current frame; shots after it are ignored.

        Returns:
            The shot type string, "-" if the player has no shots yet.
        """
        v = [s.shot_type for s in self.shots
             if s.player == player and s.frame <= frame]
        return v[-1] if v else "-"

    def movement_speed(self, player: int, frame: int) -> float:
        """Returns the player's movement speed at a frame.

        Args:
            player: Player id, 1 or 2.
            frame: Frame index.

        Returns:
            Movement speed in km/h, 0.0 if speeds were not computed.
        """
        if self.player_speeds is None:
            return 0.0
        return float(self.player_speeds[frame, player - 1])


def player_speeds(players_court: list, fps: float,
                  window_s: float = 0.5) -> np.ndarray:
    """Computes per-frame movement speed of both players.

    Speed is the displacement over a ~window_s span, which suppresses the
    frame-to-frame jitter of the projected foot point. Frames where the
    player is missing at either end of the window stay 0.

    Args:
        players_court: Per-frame dict {pid: (x, y)} in court meters.
        fps: Video frame rate.
        window_s: Time span of the displacement window, in seconds.

    Returns:
        (N, 2) speeds in km/h for players 1 and 2.
    """
    n = len(players_court)
    step = max(1, int(round(window_s * fps)))
    pos = np.full((n, 2, 2), np.nan)
    for i, fr in enumerate(players_court):
        for pid in (1, 2):
            if pid in fr:
                pos[i, pid - 1] = fr[pid]
    speeds = np.zeros((n, 2))
    a, b = pos[:-step], pos[step:]
    v = np.linalg.norm(b - a, axis=2) / (step / fps) * 3.6
    valid = np.isfinite(v)
    speeds[step:][valid] = v[valid]
    return speeds


def compute_stats(hits: list, ball_court: np.ndarray,
                  players_court: list, fps: float,
                  hit_players: dict | None = None) -> MatchStats:
    """Builds match statistics from detected hits and metric positions.

    The striker of each hit is the player closest to the ball at the
    moment of the hit; the ball speed is the straight-line distance to
    the next hit over the elapsed time.

    Args:
        hits: Shot frame indices, in increasing order.
        ball_court: (N, 2) ball positions in court meters.
        players_court: Per-frame dict {1: (x, y), 2: (x, y)} in meters.
        fps: Video frame rate.
        hit_players: Optional mapping ``frame -> player`` produced by
            image-space contact detection.

    Returns:
        A MatchStats with one ShotStat per valid hit pair and the
        per-frame player speeds.
    """
    stats = MatchStats(fps=fps,
                       player_speeds=player_speeds(players_court, fps))
    rally_pause = int(3.0 * fps)  # hit gap that starts a new rally
    search = int(0.3 * fps)       # NaN tolerance around the hit frame
    prev_hit = None
    for k, a in enumerate(hits):
        rally_start = prev_hit is None or a - prev_hit > rally_pause
        prev_hit = a

        # striker = player closest to the ball around the moment of the
        # hit (the exact hit frame may have no valid ball projection)
        ref = _nearest_valid(ball_court, a, search)
        pos = _nearest_player_positions(players_court, a, search)
        if ref is None:
            continue
        if not np.isfinite(ball_court[ref, 1]):
            continue
        striker = (
            hit_players[a] if hit_players is not None and a in hit_players
            else (1 if ball_court[ref, 1] >= NET_Y else 2)
        )
        opponent = 2 if striker == 1 else 1
        striker_pos = pos.get(striker, ball_court[ref])

        # speeds need the next hit; the last shot of a rally has none
        ball_speed, opp_speed = 0.0, 0.0
        b = hits[k + 1] if k + 1 < len(hits) else None
        if b is not None and b - a <= rally_pause:
            dt = (b - a) / fps
            if dt > 0 and np.isfinite(ball_court[[a, b]]).all():
                ball_speed = float(np.linalg.norm(
                    ball_court[b] - ball_court[a])) / dt * 3.6
                opp_a = players_court[a].get(opponent)
                opp_b = players_court[b].get(opponent)
                if opp_a is not None and opp_b is not None:
                    opp_speed = float(np.linalg.norm(
                        np.asarray(opp_b) - np.asarray(opp_a))) / dt * 3.6

        stats.shots.append(ShotStat(
            frame=a, player=striker,
            ball_speed_kmh=ball_speed,
            opponent_speed_kmh=opp_speed,
            shot_type=classify_shot(striker_pos, rally_start)))

    # a player cannot strike twice within a fraction of a second: the
    # later reversal of such a pair is a projection artifact, drop it
    min_same_player_gap = int(1.5 * fps)
    deduped = []
    for s in stats.shots:
        if (deduped and s.player == deduped[-1].player
                and s.frame - deduped[-1].frame < min_same_player_gap):
            continue
        deduped.append(s)
    stats.shots = deduped
    return stats


def _nearest_valid(arr: np.ndarray, i: int, window: int):
    """Index of the closest finite row of arr within ±window of i."""
    n = len(arr)
    for j in sorted(range(max(0, i - window), min(n, i + window + 1)),
                    key=lambda j: abs(j - i)):
        if np.isfinite(arr[j]).all():
            return j
    return None


def _nearest_player_positions(players_court: list, i: int,
                              window: int) -> dict:
    """Nearest non-empty player-position dictionary around a frame."""
    n = len(players_court)
    for j in sorted(range(max(0, i - window), min(n, i + window + 1)),
                    key=lambda frame: abs(frame - i)):
        if players_court[j]:
            return players_court[j]
    return {}
