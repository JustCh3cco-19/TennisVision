"""Conservative piecewise polynomial fit for the ball trajectory.

Between consecutive hits/bounces a tennis ball follows (in image space,
to local approximation) a smooth curve in time. Instead of allowing a
single bad detection to anchor the whole track, the smoother:

1. splits at impossible jumps, sharp direction changes and long gaps;
2. rejects nearly stationary segments (scoreboard icons, line markers);
3. fits x(t), y(t) jointly and trims points by their 2-D residual;
4. fills only gaps bounded by a supported segment.

Frames not covered by any segment stay NaN. Works in image space;
positions are projected to court space afterwards.
"""

import cv2
import numpy as np

from .geometry import COURT_MODEL_POINTS


def filter_ball_detections_to_court(
    measurements: np.ndarray,
    courts,
    visible: np.ndarray | None = None,
    outside_scale: float = 0.5,
) -> np.ndarray:
    """Rejects detections implausibly far from the projected court.

    The ball may project above the far baseline while airborne, so testing its
    homography-derived metric position would be incorrect. Instead this uses
    signed image distance from the projected court quadrilateral and permits an
    outside margin proportional to the on-screen court length.

    Args:
        measurements: (N, 2+) ball measurements; rejected rows become NaN.
        courts: Per-frame CourtReference sequence, with None on cutaways.
        visible: Optional (N,) court-visible mask.
        outside_scale: Allowed outside distance as a fraction of the mean
            projected sideline length.

    Returns:
        A filtered copy of ``measurements``.
    """
    out = np.asarray(measurements, dtype=np.float64).copy()
    if len(out) != len(courts):
        raise ValueError("measurements and courts must have equal length")
    if visible is not None and np.asarray(visible).shape != (len(out),):
        raise ValueError("visible must have shape (N,)")
    if outside_scale < 0:
        raise ValueError("outside_scale must be non-negative")

    for i in np.flatnonzero(np.isfinite(out[:, :2]).all(axis=1)):
        if ((visible is not None and not visible[i])
                or courts[i] is None):
            out[i] = np.nan
            continue
        polygon = courts[i].to_image(
            COURT_MODEL_POINTS[:4])[[0, 1, 3, 2]].astype(np.float32)
        side_length = 0.5 * (
            np.linalg.norm(polygon[3] - polygon[0])
            + np.linalg.norm(polygon[2] - polygon[1])
        )
        signed_distance = cv2.pointPolygonTest(
            polygon, tuple(out[i, :2]), measureDist=True)
        if signed_distance < -outside_scale * side_length:
            out[i] = np.nan
    return out


def filter_static_ball_detections(
    measurements: np.ndarray,
    radius_px: float = 6.0,
    window_frames: int = 30,
    min_samples: int = 5,
    min_span_frames: int = 8,
) -> np.ndarray:
    """Rejects repeated detections at a nearly fixed image location.

    Broadcast graphics, line intersections and shoes can be detected as the
    ball for many frames. A real tennis ball may cross the same small region,
    but it does not remain there over a sustained time span.

    Args:
        measurements: (N, 2+) ball measurements; rejected rows become NaN.
        radius_px: Spatial radius used to identify a static hotspot.
        window_frames: Temporal radius around each measurement.
        min_samples: Minimum nearby detections required for a hotspot.
        min_span_frames: Minimum time span covered by those detections.

    Returns:
        A filtered copy of ``measurements``.
    """
    out = np.asarray(measurements, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] < 2:
        raise ValueError("measurements must have shape (N, 2+)")
    if radius_px < 0 or window_frames < 0:
        raise ValueError("radius and window must be non-negative")
    if min_samples < 2 or min_span_frames < 1:
        raise ValueError("invalid static-hotspot thresholds")

    valid = np.flatnonzero(np.isfinite(out[:, :2]).all(axis=1))
    reject = np.zeros(len(out), dtype=bool)
    for i in valid:
        local = valid[
            (valid >= i - window_frames) & (valid <= i + window_frames)
        ]
        nearby = local[
            np.linalg.norm(out[local, :2] - out[i, :2], axis=1) <= radius_px
        ]
        if (len(nearby) >= min_samples
                and nearby[-1] - nearby[0] >= min_span_frames):
            reject[i] = True
    out[reject] = np.nan
    return out


class BallParabolicSmoother:
    """Smooths a raw ball track with per-segment parabolic fits.

    Attributes:
        gate_px: Maximum plausible displacement per frame, in pixels;
            larger jumps split the trajectory.
        max_gap: Longest run of missed detections tolerated inside a
            single segment before it is split.
        turn_cos: Cosine of the direction-change angle that splits a
            segment (hits and bounces).
        min_seg_points: Minimum valid detections for a segment to be fit.
        trim_px: Residual threshold (pixels) for the outlier-trimming
            refit pass.
        min_extent_px: Minimum 2-D range of a moving segment. Smaller
            segments are treated as static false positives.
    """

    def __init__(self, gate_px: float = 120.0,
                 max_gap: int = 12,
                 turn_angle_deg: float = 70.0,
                 min_seg_points: int = 4,
                 trim_px: float = 15.0,
                 min_extent_px: float = 10.0):
        """Initializes the smoother.

        Args:
            gate_px: Maximum plausible per-frame displacement in pixels.
            max_gap: Longest miss-run allowed inside a segment, in frames.
            turn_angle_deg: Direction change (degrees) that splits a segment.
            min_seg_points: Minimum valid points required to fit a segment.
            trim_px: Residual threshold for the refit pass, in pixels.
            min_extent_px: Minimum segment extent in pixels; rejects static
                false positives that persist at one image location.
        """
        self.gate_px = gate_px
        self.max_gap = max_gap            # longest miss-run inside a segment
        self.turn_cos = np.cos(np.deg2rad(turn_angle_deg))
        self.min_seg_points = min_seg_points
        self.trim_px = trim_px            # residual threshold for the refit
        self.min_extent_px = min_extent_px

    def smooth(self, measurements: np.ndarray) -> np.ndarray:
        """Fits and evaluates the piecewise parabolic model.

        Args:
            measurements: (N, 2) raw pixel centers, NaN for missed frames.

        Returns:
            (N, 2) fitted positions; NaN where no segment covers the frame.
        """
        pts = measurements.astype(float).copy()

        out = np.full_like(pts, np.nan)
        for idx in self._segments(pts):
            fitted = self._robust_segment_fit(idx, pts[idx])
            if fitted is None:
                continue
            span, values = fitted
            out[span] = values
        return out

    # ------------------------------------------------------------------

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
            dt = j - i
            split = ((dt - 1) > self.max_gap
                     or np.linalg.norm(pts[j] - pts[i])
                     > self.gate_px * dt)
            if not split and len(seg) >= 3:
                # multi-point baseline: single-step velocities are too
                # noisy near the apex of an arc and cause false splits
                a = seg[max(0, len(seg) - 4)]
                v_prev = (pts[i] - pts[a]) / (i - a)
                v_next = (pts[j] - pts[i]) / dt
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

    def _robust_segment_fit(self, idx: np.ndarray,
                            points: np.ndarray):
        """Fits both image coordinates with shared 2-D outlier trimming.

        Args:
            idx: Frame indices for the segment.
            points: (N, 2) measured image points at ``idx``.

        Returns:
            ``(span, values)`` for the bounded segment, or None if the
            segment is static or lacks enough mutually consistent points.
        """
        if np.linalg.norm(np.ptp(points, axis=0)) < self.min_extent_px:
            return None

        t = idx.astype(np.float64)
        center = t.mean()
        scale = max(1.0, float(np.ptp(t)) / 2.0)
        normalized = (t - center) / scale
        keep = np.ones(len(t), dtype=bool)

        for _ in range(3):
            degree = 2 if keep.sum() >= 4 else 1
            coeff_x = np.polyfit(
                normalized[keep], points[keep, 0], degree)
            coeff_y = np.polyfit(
                normalized[keep], points[keep, 1], degree)
            predicted = np.column_stack([
                np.polyval(coeff_x, normalized),
                np.polyval(coeff_y, normalized),
            ])
            residual = np.linalg.norm(predicted - points, axis=1)
            new_keep = residual <= self.trim_px
            if new_keep.sum() < self.min_seg_points:
                return None
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep

        span = np.arange(idx[0], idx[-1] + 1)
        span_normalized = (span - center) / scale
        values = np.column_stack([
            np.polyval(coeff_x, span_normalized),
            np.polyval(coeff_y, span_normalized),
        ])
        return span, values
