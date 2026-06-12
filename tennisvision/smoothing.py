"""Piecewise parabolic fit for the ball trajectory.

Between consecutive hits/bounces a tennis ball follows (in image space,
to good approximation) a parabola in time. So instead of a recursive
filter we:

1. gate outliers: detections implying an impossible jump from the
   previous accepted point are dropped (e.g. a second ball near the
   fence or a white shoe);
2. split the track into segments at sharp direction changes
   (hits and bounces) and at long detection gaps;
3. fit a robust quadratic x(t), y(t) per segment (least squares with
   one outlier-trimming refit pass) and evaluate it on every frame of
   the segment span — which also bridges missed detections inside the
   segment.

Frames not covered by any segment stay NaN. Works in image space;
positions are projected to court space afterwards.
"""

import numpy as np


class BallParabolicSmoother:
    """Smooths a raw ball track with per-segment parabolic fits.

    Attributes:
        gate_px: Maximum plausible displacement per frame, in pixels;
            detections implying a larger jump are rejected as outliers.
        max_gap: Longest run of missed detections tolerated inside a
            single segment before it is split.
        turn_cos: Cosine of the direction-change angle that splits a
            segment (hits and bounces).
        min_seg_points: Minimum valid detections for a segment to be fit.
        trim_px: Residual threshold (pixels) for the outlier-trimming
            refit pass.
    """

    def __init__(self, gate_px: float = 120.0,
                 max_gap: int = 20,
                 turn_angle_deg: float = 60.0,
                 min_seg_points: int = 4,
                 trim_px: float = 15.0):
        """Initializes the smoother.

        Args:
            gate_px: Maximum plausible per-frame displacement in pixels.
            max_gap: Longest miss-run allowed inside a segment, in frames.
            turn_angle_deg: Direction change (degrees) that splits a segment.
            min_seg_points: Minimum valid points required to fit a segment.
            trim_px: Residual threshold for the refit pass, in pixels.
        """
        self.gate_px = gate_px
        self.max_gap = max_gap            # longest miss-run inside a segment
        self.turn_cos = np.cos(np.deg2rad(turn_angle_deg))
        self.min_seg_points = min_seg_points
        self.trim_px = trim_px            # residual threshold for the refit

    def smooth(self, measurements: np.ndarray) -> np.ndarray:
        """Fits and evaluates the piecewise parabolic model.

        Args:
            measurements: (N, 2) raw pixel centers, NaN for missed frames.

        Returns:
            (N, 2) fitted positions; NaN where no segment covers the frame.
        """
        pts = measurements.astype(float).copy()
        self._reject_outliers(pts)

        out = np.full_like(pts, np.nan)
        for idx in self._segments(pts):
            t = idx.astype(float)
            cx = self._robust_quadfit(t, pts[idx, 0])
            cy = self._robust_quadfit(t, pts[idx, 1])
            # evaluate on the full frame span of the segment (fills gaps)
            span = np.arange(idx[0], idx[-1] + 1)
            out[span, 0] = np.polyval(cx, span)
            out[span, 1] = np.polyval(cy, span)
        return out

    # ------------------------------------------------------------------

    def _reject_outliers(self, pts: np.ndarray) -> None:
        last = None
        for i in range(len(pts)):
            if not np.isfinite(pts[i]).all():
                continue
            if last is not None:
                dt = i - last
                if np.linalg.norm(pts[i] - pts[last]) > self.gate_px * dt:
                    pts[i] = np.nan
                    continue
            last = i

    def _segments(self, pts: np.ndarray):
        """Splits the track into parabolic segments.

        Args:
            pts: (N, 2) gated positions, NaN for missing frames.

        Yields:
            Arrays of frame indices of valid points, split at long gaps
            and at sharp direction changes.
        """
        valid = np.flatnonzero(np.isfinite(pts).all(axis=1))
        if len(valid) == 0:
            return

        seg = [valid[0]]
        for k in range(1, len(valid)):
            i, j = seg[-1], valid[k]
            split = (j - i - 1) > self.max_gap
            if not split and len(seg) >= 3:
                # multi-point baseline: single-step velocities are too
                # noisy near the apex of an arc and cause false splits
                a = seg[max(0, len(seg) - 4)]
                v_prev = (pts[i] - pts[a]) / (i - a)
                v_next = (pts[j] - pts[i]) / (j - i)
                np_, nn = np.linalg.norm(v_prev), np.linalg.norm(v_next)
                if np_ > 1e-6 and nn > 1e-6:
                    split = (v_prev @ v_next) / (np_ * nn) < self.turn_cos
            if split:
                if len(seg) >= self.min_seg_points:
                    yield np.array(seg)
                seg = [j]
            else:
                seg.append(j)
        if len(seg) >= self.min_seg_points:
            yield np.array(seg)

    def _robust_quadfit(self, t: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Fits a quadratic with one outlier-trimming refit pass.

        Args:
            t: Sample times (frame indices) of the segment.
            v: Sample values (one pixel coordinate) at those times.

        Returns:
            Polynomial coefficients in np.polyfit order.
        """
        deg = 2 if len(t) > 2 else 1
        coeffs = np.polyfit(t, v, deg)
        resid = np.abs(np.polyval(coeffs, t) - v)
        keep = resid <= self.trim_px
        if keep.sum() >= deg + 1 and not keep.all():
            coeffs = np.polyfit(t[keep], v[keep], deg)
        return coeffs
