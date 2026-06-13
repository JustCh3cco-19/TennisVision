"""Shot (hit) and bounce detection from the ball trajectory.

A hit is a reversal of the ball's motion along the court length (y axis,
COURT coordinates): the ball travels towards one baseline, is struck, and
travels back. Working in court space (meters, perspective removed) makes
the velocity sign change much cleaner than in pixel space.

A bounce is a reversal of the ball's vertical motion in IMAGE coordinates
(falling, then rising) that happens away from both players: gravity acts
along the image vertical, so the pixel trajectory is the right space for
this signature. At the moment of a bounce the ball is on the ground, which
is the only instant where the court homography gives its exact metric
position — bounce locations are therefore the one place where in/out
geometry can be evaluated reliably from a single camera.
"""

import numpy as np

from .geometry import COURT_LENGTH, NET_Y


def detect_hits(ball_court: np.ndarray, fps: float,
                ball_px: np.ndarray = None,
                player_boxes: list = None,
                min_gap_s: float = 0.6,
                smooth_window: int = 7,
                min_speed_m_s: float = 1.0,
                box_pad_factor: float = 0.8,
                max_track_gap: int = 12,
                rally_pause_s: float = 3.0,
                baseline_margin_m: float = 4.5,
                return_players: bool = False) -> list:
    """Detects shots as reversals of the ball motion along the court.

    The projected position of a ball high above the court is displaced
    towards the far side, which can produce spurious mid-flight
    reversals. When the pixel track and the player boxes are available,
    a reversal is kept only if the ball is close to a player's bounding
    box IN THE IMAGE: at the racquet the ball overlaps the player in
    pixels regardless of its height above the court, so the image is
    the right space for this test (unlike court meters, where the
    height-induced projection offset can reach several meters).

    Args:
        ball_court: (N, 2) ball positions in court meters, NaN allowed.
        fps: Video frame rate.
        ball_px: Optional (N, 2) smoothed ball positions in pixels;
            enables the player-proximity filter.
        player_boxes: Optional per-frame dict {pid: bbox}; bbox is
            (x1, y1, x2, y2, ...) in pixels.
        min_gap_s: Minimum time between two consecutive hits, in seconds.
        smooth_window: Moving-average window (frames) applied to the y
            coordinate before differentiation.
        min_speed_m_s: Minimum |vy| required on both sides of a reversal
            for it to count as a hit.
        box_pad_factor: The player box is expanded by this fraction of
            its height on every side (the racquet reach) before testing
            whether it contains the ball.
        return_players: Return ``(frame, player)`` pairs instead of only
            frame indices.

    Returns:
        Frame indices where a shot is struck, in increasing order.
    """
    y = np.asarray(ball_court, dtype=np.float64)[:, 1]
    n = len(y)
    valid_idx = np.flatnonzero(np.isfinite(y))
    if len(valid_idx) < smooth_window * 2:
        return []

    groups = _track_groups(valid_idx, max_track_gap)
    candidates = []
    rally_pause = int(round(rally_pause_s * fps))
    previous_group_end = None

    for group_index, group in enumerate(groups):
        start, end = int(group[0]), int(group[-1])
        next_start = (
            int(groups[group_index + 1][0])
            if group_index + 1 < len(groups) else None
        )
        full_idx = np.arange(start, end + 1)
        y_filled = np.interp(full_idx, group, y[group])
        pad = smooth_window // 2
        kernel = np.ones(smooth_window) / smooth_window
        y_s = np.convolve(
            np.pad(y_filled, (pad, pad), mode="edge"),
            kernel, mode="valid")
        vy = np.gradient(y_s) * fps

        # A new track after a broadcast pause can begin with a serve, which
        # has no incoming branch and therefore no velocity reversal.
        if (previous_group_end is None
                or start - previous_group_end > rally_pause):
            contact = _contact_candidate(
                ball_court, ball_px, player_boxes, start,
                max(smooth_window, max_track_gap),
                box_pad_factor, baseline_margin_m)
            if contact is not None:
                frame, player, score = contact
                candidates.append((frame, player, score, np.inf))

        for q in range(1, len(full_idx)):
            if np.sign(vy[q]) == np.sign(vy[q - 1]) or vy[q] == 0:
                continue
            before = np.abs(vy[max(0, q - smooth_window):q]).max()
            after = np.abs(
                vy[q:min(len(vy), q + smooth_window)]).max()
            if before < min_speed_m_s or after < min_speed_m_s:
                continue
            contact = _contact_candidate(
                ball_court, ball_px, player_boxes, int(full_idx[q]),
                smooth_window, box_pad_factor, baseline_margin_m)
            if contact is not None:
                frame, player, score = contact
                candidates.append(
                    (frame, player, score, min(before, after)))

        # A fitted incoming arc and outgoing arc often leave a short NaN gap
        # exactly around racket contact. Treat only baseline/player-adjacent
        # gaps as candidates; mid-court gaps are usually missed detections.
        finite = np.isfinite(y[start:end + 1])
        runs = _boolean_runs(finite)
        for (_, left_end), (right_start, _) in zip(runs, runs[1:]):
            left = start + left_end - 1
            right = start + right_start
            gap = right - left - 1
            if gap <= 0 or gap > max_track_gap:
                continue
            contact = _contact_candidate(
                ball_court, ball_px, player_boxes, (left + right) // 2,
                max(smooth_window, gap + 2),
                box_pad_factor, baseline_margin_m)
            if contact is not None:
                frame, player, score = contact
                candidates.append((frame, player, score, 0.0))

        # If the supported track ends before a long pause or the end of the
        # clip, the last visible contact has no outgoing branch. Keep it only
        # when it is still close to a player or baseline.
        if next_start is None or next_start - end > rally_pause:
            contact = _contact_candidate(
                ball_court, ball_px, player_boxes, end,
                smooth_window, box_pad_factor, baseline_margin_m)
            if contact is not None:
                frame, player, score = contact
                candidates.append((frame, player, score, 0.0))

        previous_group_end = end

    events = _consolidate_hit_candidates(
        candidates, int(round(min_gap_s * fps)), rally_pause)
    if return_players:
        return [(int(frame), int(player))
                for frame, player, _, _ in events]
    return [int(frame) for frame, _, _, _ in events]


def _track_groups(valid_idx: np.ndarray, max_gap: int) -> list:
    """Splits valid samples where more than ``max_gap`` frames are missing."""
    if len(valid_idx) == 0:
        return []
    split = np.flatnonzero(np.diff(valid_idx) > max_gap + 1) + 1
    return [group for group in np.split(valid_idx, split) if len(group)]


def _boolean_runs(mask: np.ndarray) -> list:
    """Returns half-open runs where a boolean mask is true."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2], edges[1::2]))


def _contact_candidate(ball_court, ball_px, player_boxes, i, window,
                       pad_factor, baseline_margin):
    """Finds the most plausible contact frame and player around a candidate."""
    n = len(ball_court)
    best = None
    if ball_px is not None and player_boxes is not None:
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if not np.isfinite(ball_px[j]).all():
                continue
            for player, bbox in player_boxes[j].items():
                distance = _point_box_distance(ball_px[j], bbox)
                if best is None or distance < best[0]:
                    best = (distance, j, player)
        if best is not None and best[0] <= pad_factor:
            return best[1], best[2], best[0]

    valid = [
        j for j in range(max(0, i - window), min(n, i + window + 1))
        if np.isfinite(ball_court[j]).all()
    ]
    if not valid:
        return None
    j = min(valid, key=lambda frame: abs(frame - i))
    y = float(ball_court[j, 1])
    baseline_distance = min(abs(y), abs(COURT_LENGTH - y))
    if baseline_distance > baseline_margin:
        return None
    player = 1 if y >= NET_Y else 2
    return j, player, pad_factor + baseline_distance / baseline_margin


def _point_box_distance(point, bbox) -> float:
    """Point-to-box distance normalized by player-box height."""
    x1, y1, x2, y2 = np.asarray(bbox[:4], dtype=np.float64)
    height = max(1.0, y2 - y1)
    dx = max(x1 - point[0], 0.0, point[0] - x2)
    dy = max(y1 - point[1], 0.0, point[1] - y2)
    return float(np.hypot(dx, dy) / height)


def _consolidate_hit_candidates(candidates, min_gap, rally_pause) -> list:
    """Merges duplicate reversals and enforces player alternation."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda candidate: candidate[0])

    merged = []
    cluster = [candidates[0]]
    for candidate in candidates[1:]:
        if candidate[0] - cluster[0][0] < min_gap:
            cluster.append(candidate)
        else:
            merged.append(min(
                cluster, key=lambda item: (item[2], -item[3])))
            cluster = [candidate]
    merged.append(min(cluster, key=lambda item: (item[2], -item[3])))

    alternating = []
    for candidate in merged:
        if (alternating
                and candidate[0] - alternating[-1][0] <= rally_pause
                and candidate[1] == alternating[-1][1]):
            if (candidate[2], -candidate[3]) < (
                    alternating[-1][2], -alternating[-1][3]):
                alternating[-1] = candidate
            continue
        alternating.append(candidate)
    return alternating


def detect_bounces(ball_px: np.ndarray, ball_court: np.ndarray,
                   hits: list, players_court: list, fps: float,
                   smooth_window: int = 5,
                   min_speed_px_s: float = 35.0,
                   hit_exclude_s: float = 0.3,
                   player_exclude_m: float = 2.5,
                   min_gap_s: float = 0.3) -> list:
    """Detects ground bounces of the ball.

    A bounce is a falling-to-rising reversal of the vertical pixel
    velocity (image y grows downwards) that is not explained by a player
    striking the ball: candidates close in time to a detected hit, or
    close in court space to either player, are rejected.

    Args:
        ball_px: (N, 2) smoothed ball positions in image pixels.
        ball_court: (N, 2) ball positions projected to court meters.
        hits: Shot frame indices from detect_hits().
        players_court: Per-frame dict {pid: (x, y)} in court meters.
        fps: Video frame rate.
        smooth_window: Moving-average window (frames) on the vertical
            pixel coordinate before differentiation.
        min_speed_px_s: Minimum |vertical velocity| on both sides of the
            reversal, in pixels per second.
        hit_exclude_s: Half-width of the exclusion window around each
            detected hit, in seconds.
        player_exclude_m: Court-space distance to the nearest player
            below which a reversal is attributed to a hit, not a bounce.
        min_gap_s: Minimum time between two consecutive bounces.

    Returns:
        Frame indices of the detected bounces, in increasing order.
    """
    y = ball_px[:, 1].copy()
    n = len(y)
    valid = np.isfinite(y)
    idx = np.arange(n)
    if valid.sum() < smooth_window * 2:
        return []
    y_filled = np.interp(idx, idx[valid], y[valid])
    kernel = np.ones(smooth_window) / smooth_window
    y_s = np.convolve(y_filled, kernel, mode="same")
    vy = np.gradient(y_s) * fps  # px/s, positive while falling

    hit_excl = int(hit_exclude_s * fps)
    min_gap = int(min_gap_s * fps)
    bounces = []
    for i in range(1, n):
        if not (vy[i - 1] > 0 and vy[i] <= 0):     # falling -> rising
            continue
        if not valid[max(0, i - smooth_window):
                     min(n, i + smooth_window)].any():
            continue                                # inside a long gap
        before = vy[max(0, i - smooth_window):i].max()
        after = -vy[i:min(n, i + smooth_window)].min()
        if before < min_speed_px_s or after < min_speed_px_s:
            continue
        if any(abs(i - h) <= hit_excl for h in hits):
            continue
        if np.isfinite(ball_court[i]).all():
            near = min((np.linalg.norm(np.asarray(p) - ball_court[i])
                        for p in players_court[i].values()), default=np.inf)
            if near < player_exclude_m:
                continue
        if bounces and i - bounces[-1] < min_gap:
            continue
        bounces.append(i)
    return bounces
