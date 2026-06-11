"""Shot (hit) detection from the ball trajectory in COURT coordinates.

A hit is a reversal of the ball's motion along the court length (y axis):
the ball travels towards one baseline, is struck, and travels back. Working
in court space (meters, perspective removed) makes the velocity sign change
much cleaner than in pixel space. Detected automatically — no precomputed
stubs.
"""

import numpy as np


def detect_hits(ball_court: np.ndarray, fps: float,
                min_gap_s: float = 0.6,
                smooth_window: int = 7,
                min_speed_m_s: float = 1.0) -> list:
    """ball_court: (N,2) ball positions in meters (NaN allowed).
    Returns the list of frame indices where a shot is struck."""
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
            hits.append(i)
    return hits
