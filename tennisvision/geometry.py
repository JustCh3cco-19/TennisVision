"""Court geometry: canonical metric model of a tennis court and the
homography that links image pixels to real-world court coordinates.

All real-world coordinates are expressed in meters, with the origin in the
top-left corner of the doubles court, x growing to the right and y growing
towards the bottom baseline (same convention as image coordinates, which
keeps the homography well-behaved).
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Official ITF dimensions (meters)
COURT_WIDTH_DOUBLES = 10.97
COURT_WIDTH_SINGLES = 8.23
COURT_LENGTH = 23.77
ALLEY_WIDTH = (COURT_WIDTH_DOUBLES - COURT_WIDTH_SINGLES) / 2  # 1.37
SERVICE_FROM_NET = 6.40
NET_Y = COURT_LENGTH / 2  # 11.885

_SL = ALLEY_WIDTH                      # singles left x
_SR = COURT_WIDTH_DOUBLES - ALLEY_WIDTH  # singles right x
_TOP_SERVICE_Y = NET_Y - SERVICE_FROM_NET
_BOT_SERVICE_Y = NET_Y + SERVICE_FROM_NET
_CX = COURT_WIDTH_DOUBLES / 2

# Canonical 14-keypoint model. The court keypoint detector must be trained
# with the SAME ordering (see training/prepare_court_pose_dataset.py).
COURT_MODEL_POINTS = np.array(
    [
        (0.0, 0.0),                          # 0  doubles corner, top-left
        (COURT_WIDTH_DOUBLES, 0.0),          # 1  doubles corner, top-right
        (0.0, COURT_LENGTH),                 # 2  doubles corner, bottom-left
        (COURT_WIDTH_DOUBLES, COURT_LENGTH), # 3  doubles corner, bottom-right
        (_SL, 0.0),                          # 4  singles corner, top-left
        (_SR, 0.0),                          # 5  singles corner, top-right
        (_SL, COURT_LENGTH),                 # 6  singles corner, bottom-left
        (_SR, COURT_LENGTH),                 # 7  singles corner, bottom-right
        (_SL, _TOP_SERVICE_Y),               # 8  top service line, left
        (_SR, _TOP_SERVICE_Y),               # 9  top service line, right
        (_SL, _BOT_SERVICE_Y),               # 10 bottom service line, left
        (_SR, _BOT_SERVICE_Y),               # 11 bottom service line, right
        (_CX, _TOP_SERVICE_Y),               # 12 top "T"
        (_CX, _BOT_SERVICE_Y),               # 13 bottom "T"
    ],
    dtype=np.float64,
)

# Line segments of the court, as index pairs into COURT_MODEL_POINTS,
# used to render the top-down minimap.
COURT_LINES = [
    (0, 1), (2, 3),          # baselines
    (0, 2), (1, 3),          # doubles sidelines
    (4, 6), (5, 7),          # singles sidelines
    (8, 9), (10, 11),        # service lines
    (12, 13),                # center service line (drawn full, clipped visually by net)
]


def smooth_court_keypoints(keypoints_frames: np.ndarray,
                           visible: np.ndarray,
                           window: int = 11) -> np.ndarray:
    """Temporally stabilizes per-frame court keypoints with a local median.

    Smoothing is performed independently inside each contiguous court-visible
    segment, so detections from a replay or close-up cannot leak into the next
    main-camera shot. A centered window removes pose jitter while still
    following broadcast-camera pans and zooms.

    Args:
        keypoints_frames: (N, K, 2) detected image keypoints.
        visible: (N,) mask of frames considered valid court views.
        window: Odd temporal median window size in frames.

    Returns:
        (N, K, 2) stabilized keypoints. Non-visible frames remain NaN.

    Raises:
        ValueError: If the inputs have incompatible shapes or the window is
            not a positive odd integer.
    """
    points = np.asarray(keypoints_frames, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError("keypoints_frames must have shape (N, K, 2)")
    if visible.shape != (len(points),):
        raise ValueError("visible must have shape (N,)")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    out = np.full_like(points, np.nan)
    radius = window // 2
    edges = np.flatnonzero(np.diff(np.r_[False, visible, False]))
    for start, end in zip(edges[::2], edges[1::2]):
        for i in range(start, end):
            lo = max(start, i - radius)
            hi = min(end, i + radius + 1)
            local = points[lo:hi]
            for keypoint in range(points.shape[1]):
                for axis in range(2):
                    values = local[:, keypoint, axis]
                    values = values[np.isfinite(values)]
                    if len(values):
                        out[i, keypoint, axis] = np.median(values)
    return out


def _court_line_samples(samples_per_meter: float = 4.0) -> np.ndarray:
    """Densely samples points along the painted court lines, in meters.

    Used as the source set for chamfer-ICP homography refinement. The net is
    deliberately excluded: it is off the court plane and has no painted line.

    Args:
        samples_per_meter: Sampling density along each line segment.

    Returns:
        (M, 2) points in court meters.
    """
    segments = []
    for a, b in COURT_LINES:
        pa, pb = COURT_MODEL_POINTS[a], COURT_MODEL_POINTS[b]
        n = max(2, int(np.hypot(*(pb - pa)) * samples_per_meter))
        t = np.linspace(0.0, 1.0, n)[:, None]
        segments.append(pa[None] * (1.0 - t) + pb[None] * t)
    return np.concatenate(segments, axis=0)


@dataclass
class CourtReference:
    """Image<->court mapping estimated from the detected keypoints."""

    homography: np.ndarray          # image px -> court meters (3x3)
    inverse: np.ndarray             # court meters -> image px
    keypoints_px: np.ndarray        # (N, 2) detected image points
    inliers: np.ndarray             # (N,) bool: RANSAC inlier per keypoint
    #                                 (False for outliers and missed detections)

    @classmethod
    def from_keypoints(cls, keypoints_px: np.ndarray,
                       min_points: int = 4,
                       ransac_thresh: float = 0.4) -> "CourtReference":
        """Fits a homography with RANSAC from detected keypoints.

        Accepts either the full 14-keypoint set or any prefix of
        COURT_MODEL_POINTS (e.g. just the 4 doubles corners). Points with
        non-finite coordinates (missed detections) are dropped; RANSAC
        tolerates residual localization noise on the rest.

        Args:
            keypoints_px: (N, 2) detected keypoints in image pixels, in
                COURT_MODEL_POINTS order. NaN rows mark missed detections.
            min_points: Minimum number of finite keypoints required.
            ransac_thresh: RANSAC reprojection threshold, in court meters
                (destination space of the homography).

        Returns:
            A CourtReference with the fitted homography and its inverse.

        Raises:
            ValueError: If fewer than ``min_points`` keypoints are finite,
                or the homography estimation fails.
        """
        keypoints_px = np.asarray(keypoints_px, dtype=np.float64).reshape(-1, 2)
        model_points = COURT_MODEL_POINTS[:len(keypoints_px)]
        valid = np.isfinite(keypoints_px).all(axis=1)
        if valid.sum() < min_points:
            raise ValueError(
                f"only {int(valid.sum())} valid court keypoints, "
                f"need at least {min_points}")
        H, mask = cv2.findHomography(
            keypoints_px[valid], model_points[valid],
            method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if H is None:
            raise ValueError("homography estimation failed")
        # Map the RANSAC inlier mask (defined over the valid subset) back
        # onto the full keypoint array; missed detections are not inliers.
        inliers = np.zeros(len(keypoints_px), dtype=bool)
        inliers[valid] = mask.ravel().astype(bool)
        return cls(homography=H, inverse=np.linalg.inv(H),
                   keypoints_px=keypoints_px, inliers=inliers)

    def refine_to_lines(self, line_mask: np.ndarray, n_iter: int = 12,
                        max_dist_px: float = 40.0, min_matches: int = 30,
                        samples_per_meter: float = 4.0,
                        region_margin: float = 1.5) -> "CourtReference":
        """Refines the homography by snapping the model lines to the image lines.

        The keypoint-based fit gets the court shape right but, on out-of-domain
        surfaces (clay, grass), can be globally biased: the projected overlay
        keeps its shape yet drifts off the real lines. This corrects that
        residual error directly in image space, without retraining, by chamfer
        ICP: the canonical court line skeleton is projected into the image,
        each sample point is matched to the nearest detected white-line pixel,
        and a new homography is re-fit from those correspondences. Iterating
        with a shrinking match radius pulls the overlay onto the painted lines.

        Only the painted ground lines are used (the net is off-plane and has no
        painted line, so it is excluded). The keypoint fit supplies the
        initialization, so the nearest white pixel is almost always on the
        correct line; a robust (LMEDS) re-fit rejects the occasional mismatch.

        Args:
            line_mask: Boolean (H, W) mask of white-line pixels, e.g. from
                ``white_line_mask`` on the frame being refined.
            n_iter: ICP iterations.
            max_dist_px: Initial match radius in pixels; shrinks across
                iterations to refine the alignment.
            min_matches: Minimum correspondences required to re-fit; below it
                the refinement stops and the current estimate is kept.
            samples_per_meter: Sampling density along the court lines.
            region_margin: Margin (meters) around the court used to mask the
                line pixels, so crowd/advertising whites cannot be matched.

        Returns:
            A new CourtReference with the refined homography, or ``self``
            unchanged if the mask is empty or the refinement does not converge
            to a geometrically sane result.
        """
        from scipy import ndimage

        mask = np.asarray(line_mask, dtype=bool)
        if mask.ndim != 2 or mask.sum() < min_matches:
            return self
        h_img, w_img = mask.shape

        # keep only line pixels inside the projected court (+margin): the crowd,
        # scoreboard and sponsor boards are white too and would corrupt ICP.
        region = self._court_region_mask(h_img, w_img, region_margin)
        mask = mask & region
        if mask.sum() < min_matches:
            return self

        # distance transform: for every pixel, the distance to and index of the
        # nearest white-line pixel (chamfer matching in O(1) per query).
        dist, (iy, ix) = ndimage.distance_transform_edt(
            ~mask, return_indices=True)
        court_pts = _court_line_samples(samples_per_meter)

        inv = self.inverse.copy()  # court meters -> image px (refined in place)
        for it in range(n_iter):
            radius = max_dist_px * (1.0 - 0.6 * it / max(1, n_iter - 1))
            proj = self._apply(inv, court_pts)
            px = np.round(proj[:, 0]).astype(np.int64)
            py = np.round(proj[:, 1]).astype(np.int64)
            inside = (px >= 0) & (px < w_img) & (py >= 0) & (py < h_img)
            if inside.sum() < min_matches:
                break
            d = dist[py[inside], px[inside]]
            keep = d < radius
            if keep.sum() < min_matches:
                break
            src = court_pts[inside][keep]
            yy = py[inside][keep]
            xx = px[inside][keep]
            dst = np.stack([ix[yy, xx], iy[yy, xx]], axis=1).astype(np.float64)
            inv_new, _ = cv2.findHomography(src, dst, method=cv2.LMEDS)
            if inv_new is None:
                break
            inv = inv_new

        H = np.linalg.inv(inv)
        refined = CourtReference(homography=H, inverse=inv,
                                 keypoints_px=self.keypoints_px,
                                 inliers=self.inliers)
        if not self._is_sane_refinement(refined, w_img, h_img):
            return self  # fall back to the keypoint fit rather than risk worse
        return refined

    def _court_region_mask(self, h_img: int, w_img: int,
                           margin: float) -> np.ndarray:
        """Boolean image mask of the court area (+margin), via the current fit."""
        corners_m = np.array([
            [-margin, -margin],
            [COURT_WIDTH_DOUBLES + margin, -margin],
            [COURT_WIDTH_DOUBLES + margin, COURT_LENGTH + margin],
            [-margin, COURT_LENGTH + margin],
        ])
        poly = self.to_image(corners_m)
        out = np.zeros((h_img, w_img), dtype=np.uint8)
        if np.isfinite(poly).all():
            cv2.fillConvexPoly(out, np.round(poly).astype(np.int32), 1)
        return out.astype(bool)

    def _is_sane_refinement(self, refined: "CourtReference",
                            w_img: int, h_img: int) -> bool:
        """Rejects a refined homography that is degenerate or wildly displaced.

        ICP can diverge if too many line points latch onto the wrong line. The
        refinement is only accepted if the projected court corners stay convex,
        keep a comparable area, and do not jump by more than a quarter of the
        image diagonal from the keypoint-based estimate.
        """
        corners_m = np.array([[0.0, 0.0], [COURT_WIDTH_DOUBLES, 0.0],
                              [COURT_WIDTH_DOUBLES, COURT_LENGTH],
                              [0.0, COURT_LENGTH]])
        before = self.to_image(corners_m)
        after = refined.to_image(corners_m)
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            return False
        diag = np.hypot(w_img, h_img)
        if np.max(np.linalg.norm(after - before, axis=1)) > 0.25 * diag:
            return False
        area_before = abs(cv2.contourArea(before.astype(np.float32)))
        area_after = abs(cv2.contourArea(after.astype(np.float32)))
        if area_before <= 0 or not 0.5 <= area_after / area_before <= 2.0:
            return False
        return cv2.isContourConvex(after.astype(np.float32).reshape(-1, 1, 2))

    def to_court(self, points_px: np.ndarray) -> np.ndarray:
        """Projects image points to court coordinates.

        Args:
            points_px: (N, 2) points in image pixels.

        Returns:
            (N, 2) points in court meters.
        """
        return self._apply(self.homography, points_px)

    def to_image(self, points_m: np.ndarray) -> np.ndarray:
        """Projects court points back to image pixels.

        Args:
            points_m: (N, 2) points in court meters.

        Returns:
            (N, 2) points in image pixels.
        """
        return self._apply(self.inverse, points_m)

    @staticmethod
    def _apply(H: np.ndarray, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, H)
        return out.reshape(-1, 2)

    def contains(self, point_m: np.ndarray, margin: float = 2.0) -> bool:
        """Checks whether a court-space point lies on the court.

        Args:
            point_m: (x, y) position in court meters.
            margin: Tolerance around the court boundary, in meters.

        Returns:
            True if the point is on the court (within the margin).
        """
        x, y = point_m
        return (-margin <= x <= COURT_WIDTH_DOUBLES + margin
                and -margin <= y <= COURT_LENGTH + margin)
