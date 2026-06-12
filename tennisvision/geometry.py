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


@dataclass
class CourtReference:
    """Image<->court mapping estimated from the detected keypoints."""

    homography: np.ndarray          # image px -> court meters (3x3)
    inverse: np.ndarray             # court meters -> image px
    keypoints_px: np.ndarray        # (N, 2) detected image points

    @classmethod
    def from_keypoints(cls, keypoints_px: np.ndarray,
                       min_points: int = 4) -> "CourtReference":
        """Fits a homography with RANSAC from detected keypoints.

        Accepts either the full 14-keypoint set or any prefix of
        COURT_MODEL_POINTS (e.g. just the 4 doubles corners). Points with
        non-finite coordinates (missed detections) are dropped; RANSAC
        tolerates residual localization noise on the rest.

        Args:
            keypoints_px: (N, 2) detected keypoints in image pixels, in
                COURT_MODEL_POINTS order. NaN rows mark missed detections.
            min_points: Minimum number of finite keypoints required.

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
        H, _ = cv2.findHomography(
            keypoints_px[valid], model_points[valid],
            method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if H is None:
            raise ValueError("homography estimation failed")
        return cls(homography=H, inverse=np.linalg.inv(H),
                   keypoints_px=keypoints_px)

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
