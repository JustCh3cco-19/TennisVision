"""Tennis ball detection with a fine-tuned YOLO26 model.

Returns the raw per-frame measurement (or NaN when the ball is missed);
gap filling and smoothing are delegated to the piecewise parabolic
fit in tennisvision.smoothing, NOT done here.
"""

import numpy as np
from ultralytics import YOLO


class BallDetector:
    def __init__(self, model_path: str, conf: float = 0.15):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_frames(self, frames) -> np.ndarray:
        """Returns an (N, 2) array of ball centers in pixels, NaN if missed.

        Accepts any iterable of frames (list or generator). When multiple
        candidates are detected, the most confident one is kept.
        """
        centers = []
        for frame in frames:
            center = (np.nan, np.nan)
            result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
            if result.boxes is not None and len(result.boxes) > 0:
                best = int(result.boxes.conf.argmax())
                x1, y1, x2, y2 = result.boxes.xyxy[best].tolist()
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            centers.append(center)
        return np.array(centers, dtype=np.float64)
