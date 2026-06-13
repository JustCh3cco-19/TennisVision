"""Court keypoint detection with YOLO26-pose.

The court is treated as a single "object" with 14 keypoints (see
tennisvision.geometry.COURT_MODEL_POINTS for the ordering). Keypoints below
the visibility threshold are returned as NaN and excluded from the
homography fit by CourtReference.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from ..progress import Progress

N_KEYPOINTS = 14


def clahe_normalize(frame: np.ndarray) -> np.ndarray:
    """Contrast-normalizes a BGR frame with CLAHE on the L channel.

    The court model was trained mostly on well-lit hardcourt footage and
    degrades on clay/grass under natural light, where the surface color and
    harsh shadows are out of distribution. Applying CLAHE (Contrast Limited
    Adaptive Histogram Equalization) to the luminance channel evens out the
    lighting and boosts the contrast of the court lines without retraining,
    pushing the frame closer to the distribution the model handles well.

    Args:
        frame: BGR frame.

    Returns:
        A contrast-normalized BGR frame (same shape and dtype).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def white_line_mask(frame: np.ndarray, kernel_size: int = 17,
                    tophat_thresh: float = 18.0) -> np.ndarray:
    """Segments the painted court lines from a single frame.

    A plain brightness/colour threshold fails on clay: the lines pick up red
    dust (so they are not neutral) and are dimmer than the sunlit stands,
    scoreboard and sponsor boards, which dominate any global threshold. Court
    lines are instead defined by their *shape* — thin bright ridges against a
    locally darker, uniform surface — which a white top-hat isolates regardless
    of surface colour or absolute brightness: the morphological opening erases
    structures thinner than the kernel (the lines), and subtracting it leaves
    exactly those lines. Large bright regions (stands, boards) survive the
    opening, so the top-hat suppresses them.

    Args:
        frame: BGR frame.
        kernel_size: Opening kernel size in pixels; must exceed the on-screen
            line width so the lines are removed by the opening (and thus kept
            by the top-hat).
        tophat_thresh: Minimum top-hat response (0-255) for a line pixel.

    Returns:
        Boolean mask, True on court-line pixels.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(l, cv2.MORPH_TOPHAT, kernel)
    return tophat >= tophat_thresh


def static_line_mask(frames, min_fraction: float = 0.5) -> np.ndarray:
    """Aggregates per-frame white masks into a static-lines mask.

    The painted lines are fixed in a static-camera broadcast; players, ball
    and shadows move. Keeping only pixels that read as white in at least
    ``min_fraction`` of the sampled frames isolates the lines and discards the
    transient white of shirts, socks or the ball passing through.

    Args:
        frames: Iterable of BGR frames (sparsely sampled court-visible frames).
        min_fraction: Fraction of frames a pixel must be white in to be kept.

    Returns:
        Boolean mask, True on persistent white-line pixels. Empty array if no
        frames are provided.
    """
    acc = None
    n = 0
    for frame in frames:
        m = white_line_mask(frame).astype(np.float32)
        acc = m if acc is None else acc + m
        n += 1
    if acc is None:
        return np.zeros((0, 0), dtype=bool)
    return (acc / n) >= min_fraction


class CourtKeypointDetector:
    """Court keypoint detector wrapping a trained YOLO26-pose model."""

    def __init__(self, model_path: str, kpt_conf: float = 0.5,
                 preprocess: bool = False):
        """Loads the model.

        Args:
            model_path: Path to the trained court-pose weights (.pt).
            kpt_conf: Per-keypoint confidence threshold; keypoints below
                it are returned as NaN.
            preprocess: If True, CLAHE-normalize each frame before
                inference (helps on clay/grass and natural light).
        """
        self.model = YOLO(model_path)
        self.kpt_conf = kpt_conf
        self.preprocess = preprocess

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detects the court keypoints on a single frame.

        Args:
            frame: BGR frame.

        Returns:
            (14, 2) keypoints in pixels, NaN where not confident.

        Raises:
            ValueError: If no court is detected in the frame.
        """
        if self.preprocess:
            frame = clahe_normalize(frame)
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

    def detect_frames(self, frames, total: int | None = None) -> np.ndarray:
        """Detects court keypoints on every frame.

        Args:
            frames: Any iterable of BGR frames (list or generator).
            total: Expected frame count, used for terminal progress.

        Returns:
            (N, 14, 2) per-frame keypoints; all-NaN rows for frames where
            no court is detected (close-ups, replays, crowd shots).
        """
        out = []
        progress = Progress("court keypoints", total)
        for i, frame in enumerate(frames):
            try:
                out.append(self.detect(frame))
            except ValueError:
                out.append(np.full((N_KEYPOINTS, 2), np.nan))
            progress.update(i + 1)
        progress.close(len(out))
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
