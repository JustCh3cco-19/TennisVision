"""Tennis ball detection with a fine-tuned YOLO26 model.

Returns the raw per-frame measurement (or NaN when the ball is missed);
gap filling and smoothing are delegated to the piecewise parabolic
fit in tennisvision.smoothing, NOT done here.
"""

import numpy as np
from ultralytics import YOLO


class BallDetector:
    """Single-class ball detector wrapping a fine-tuned YOLO26 model."""

    def __init__(self, model_path: str, conf: float = 0.15):
        """Loads the model.

        Args:
            model_path: Path to the fine-tuned ball weights (.pt).
            conf: Detection confidence threshold; kept low on purpose,
                outlier rejection is delegated to the smoother.
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_frames(self, frames) -> np.ndarray:
        """Detects the ball on every frame.

        When multiple candidates are detected, the most confident one is
        kept.

        Args:
            frames: Any iterable of BGR frames (list or generator).

        Returns:
            (N, 3) array of (x, y, conf) per frame in pixels; NaN rows
            for frames where the ball is missed.
        """
        centers = []
        for i, frame in enumerate(frames):
            if i % 200 == 0:
                print(f"\r  ball detection: frame {i}", end="", flush=True)
            center = (np.nan, np.nan, np.nan)
            result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                best = int(result.boxes.conf.argmax())
                x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                          float(result.boxes.conf[best]))
            centers.append(center)
        print()
        return np.array(centers, dtype=np.float64)
