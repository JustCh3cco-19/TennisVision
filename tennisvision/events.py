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


def detect_hits(ball_court: np.ndarray, fps: float,
                ball_px: np.ndarray = None,
                player_boxes: list = None,
                min_gap_s: float = 0.6,
                smooth_window: int = 7,
                min_speed_m_s: float = 1.0,
                box_pad_factor: float = 0.8) -> list:
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

    Returns:
        Frame indices where a shot is struck, in increasing order.
    """
    y = ball_court[:, 1].copy()
    n = len(y)

    # short moving-average on the valid samples to suppress jitter
    valid = np.isfinite(y)
    idx = np.arange(n)
    if valid.sum() < smooth_window * 2:
        return []
    y_filled = np.interp(idx, idx[valid], y[valid])
    kernel = np.ones(smooth_window) / smooth_window
    y_s = np.convolve(y_filled, kernel, mode="same")

    vy = np.gradient(y_s) * fps  # m/s along the court

    hits = []
    min_gap = int(min_gap_s * fps)
    for i in range(1, n):
        if np.sign(vy[i]) != np.sign(vy[i - 1]) and vy[i] != 0:
            # require real motion on both sides of the reversal
            before = np.abs(vy[max(0, i - smooth_window):i]).max()
            after = np.abs(vy[i:min(n, i + smooth_window)]).max()
            if before < min_speed_m_s or after < min_speed_m_s:
                continue
            if hits and i - hits[-1] < min_gap:
                continue
            if ball_px is not None and player_boxes is not None:
                if not _near_player_box(ball_px, player_boxes, i,
                                        smooth_window, box_pad_factor):
                    continue
            hits.append(i)
    return hits


def _near_player_box(ball_px, player_boxes, i, window, pad_factor) -> bool:
    """True if, around frame i, the ball lies in an expanded player box."""
    n = len(ball_px)
    for j in range(max(0, i - window), min(n, i + window + 1)):
        if not np.isfinite(ball_px[j]).all():
            continue
        bx, by = ball_px[j]
        for bbox in player_boxes[j].values():
            x1, y1, x2, y2 = bbox[:4]
            pad = pad_factor * (y2 - y1)
            if (x1 - pad <= bx <= x2 + pad
                    and y1 - pad <= by <= y2 + pad):
                return True
    return False


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
