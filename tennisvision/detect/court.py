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
    def __init__(self, model_path: str, kpt_conf: float = 0.5):
        self.model = YOLO(model_path)
        self.kpt_conf = kpt_conf

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Returns (14, 2) keypoints in pixels, NaN where not confident."""
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
        """Per-frame keypoints: (N, 14, 2), all-NaN rows for frames where no
        court is detected (close-ups, replays, crowd shots).

        Accepts any iterable of frames (list or generator).
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
        """Median of keypoints over sampled frames where the court is visible:
        cheap temporal aggregation that suppresses per-frame jitter.

        `video` is a VideoReader (sparse seeks, no full decode in memory).
        Broadcast footage cuts away from the court view (close-ups, crowd,
        replays), so frames where no court is detected are skipped; sampling
        stops after `n_samples` successful detections.
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
