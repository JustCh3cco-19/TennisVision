"""Video input/output helpers.

VideoReader streams frames from disk instead of holding the decoded video
in memory: a full broadcast match at 1080p does not fit in RAM (~9 GB per
minute), so the pipeline iterates over the file once per pass and keeps
only the (tiny) per-frame detections.
"""

from dataclasses import dataclass

import cv2
import numpy as np


class VideoReader:
    """Streaming, re-iterable access to a video file."""

    def __init__(self, path: str):
        self.path = path
        cap = self._open()
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if self.n_frames <= 0:
            raise ValueError(f"no frames in {path}")

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        return cap

    def frames(self):
        """Generator over all frames (BGR). Each call re-reads the file."""
        cap = self._open()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    def frame_at(self, index: int) -> np.ndarray:
        """Random access to a single frame (seek, used for sparse sampling)."""
        cap = self._open()
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"cannot read frame {index} of {self.path}")
            return frame
        finally:
            cap.release()


class VideoWriter:
    """Incremental writer: frames go to disk as they are rendered."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass
class VideoClip:
    """In-memory clip; only suitable for short videos."""
    frames: list  # list[np.ndarray] BGR
    fps: float

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def load_video(path: str) -> VideoClip:
    reader = VideoReader(path)
    return VideoClip(frames=list(reader.frames()), fps=reader.fps)


def save_video(frames: list, fps: float, path: str) -> None:
    h, w = frames[0].shape[:2]
    with VideoWriter(path, fps, w, h) as writer:
        for frame in frames:
            writer.write(frame)
