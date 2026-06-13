"""Tennis ball detection with a fine-tuned YOLO26 model.

Returns the raw per-frame measurement (or NaN when the ball is missed);
gap filling and smoothing are delegated to the piecewise parabolic
fit in tennisvision.smoothing, NOT done here.
"""

import numpy as np
from ultralytics import YOLO

from ..progress import Progress


class BallDetector:
    """Single-class ball detector wrapping a fine-tuned YOLO26 model."""

    def __init__(self, model_path: str, conf: float = 0.15,
                 imgsz: int = 1280):
        """Loads the model.

        Args:
            model_path: Path to the fine-tuned ball weights (.pt).
            conf: Detection confidence threshold; kept low on purpose,
                outlier rejection is delegated to the smoother.
            imgsz: Inference resolution. A tennis ball is only a few pixels
                wide in a 1080p broadcast; 1280 preserves substantially more
                detail than the Ultralytics default of 640, especially on
                grass and clay.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect_frames(self, frames, total: int | None = None) -> np.ndarray:
        """Detects the ball on every frame.

        When multiple candidates are detected, the most confident one is
        kept.

        Args:
            frames: Any iterable of BGR frames (list or generator).
            total: Expected frame count, used for terminal progress.

        Returns:
            (N, 3) array of (x, y, conf) per frame in pixels; NaN rows
            for frames where the ball is missed.
        """
        centers = []
        progress = Progress("ball detection", total)
        for i, frame in enumerate(frames):
            center = (np.nan, np.nan, np.nan)
            result = self.model.predict(
                frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                best = int(result.boxes.conf.argmax())
                x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                          float(result.boxes.conf[best]))
            centers.append(center)
            progress.update(i + 1)
        progress.close(len(centers))
        return np.array(centers, dtype=np.float64)
