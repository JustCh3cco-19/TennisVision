"""Court keypoint detection with YOLO26-pose.

The court is treated as a single "object" with 14 keypoints (see
tennisvision.geometry.COURT_MODEL_POINTS for the ordering). Keypoints below
the visibility threshold are returned as NaN and excluded from the
homography fit by CourtReference.
"""

import numpy as np
from ultralytics import YOLO

N_KEYPOINTS = 14


class CourtKeypointDetector:
    """Court keypoint detector wrapping a trained YOLO26-pose model."""

    def __init__(self, model_path: str, kpt_conf: float = 0.5):
        """Loads the model.

        Args:
            model_path: Path to the trained court-pose weights (.pt).
            kpt_conf: Per-keypoint confidence threshold; keypoints below
                it are returned as NaN.
        """
        self.model = YOLO(model_path)
        self.kpt_conf = kpt_conf

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detects the court keypoints on a single frame.

        Args:
            frame: BGR frame.

        Returns:
            (14, 2) keypoints in pixels, NaN where not confident.

        Raises:
            ValueError: If no court is detected in the frame.
        """
        result = self.model.predict(frame, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            raise ValueError("no court detected in frame")
        kpts = result.keypoints.xy[0].cpu().numpy()
        confs = (result.keypoints.conf[0].cpu().numpy()
                 if result.keypoints.conf is not None
                 else np.ones(len(kpts)))
        kpts = kpts[:N_KEYPOINTS].astype(np.float64)
        kpts[confs[:N_KEYPOINTS] < self.kpt_conf] = np.nan
        return kpts

    def detect_frames(self, frames) -> np.ndarray:
        """Detects court keypoints on every frame.

        Args:
            frames: Any iterable of BGR frames (list or generator).

        Returns:
            (N, 14, 2) per-frame keypoints; all-NaN rows for frames where
            no court is detected (close-ups, replays, crowd shots).
        """
        out = []
        for frame in frames:
            try:
                out.append(self.detect(frame))
            except ValueError:
                out.append(np.full((N_KEYPOINTS, 2), np.nan))
        return np.stack(out)

    def detect_median(self, video, n_samples: int = 5,
                      max_attempts: int = 30) -> np.ndarray:
        """Aggregates keypoints over sparsely sampled frames.

        Broadcast footage cuts away from the court view (close-ups, crowd,
        replays), so frames where no court is detected are skipped; the
        median over the successful detections suppresses per-frame jitter.

        Args:
            video: A VideoReader (sparse seeks, no full decode in memory).
            n_samples: Successful detections to aggregate before stopping.
            max_attempts: Frames sampled uniformly across the video.

        Returns:
            (14, 2) median keypoints in pixels.

        Raises:
            ValueError: If no court is detected in any sampled frame.
        """
        idx = np.linspace(0, video.n_frames - 1, max_attempts, dtype=int)
        detections = []
        for i in idx:
            try:
                detections.append(self.detect(video.frame_at(int(i))))
            except ValueError:
                continue  # no court in this frame (cutaway) or unreadable
            if len(detections) >= n_samples:
                break
        if not detections:
            raise ValueError(
                f"no court detected in any of {max_attempts} sampled frames")
        return np.nanmedian(np.stack(detections), axis=0)
