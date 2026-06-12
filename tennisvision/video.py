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
        """Opens the file once to read its metadata.

        Args:
            path: Path to the video file.

        Raises:
            FileNotFoundError: If the file cannot be opened.
            ValueError: If the file contains no frames.
        """
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
        """Iterates over all frames; each call re-reads the file.

        Yields:
            BGR frames in playback order.
        """
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
        """Random access to a single frame (seek, used for sparse sampling).

        Args:
            index: Frame index.

        Returns:
            The BGR frame at that index.

        Raises:
            ValueError: If the frame cannot be read.
        """
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
        """Opens the output file for writing.

        Args:
            path: Output video path (.mp4).
            fps: Output frame rate.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
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
    """Loads a whole video into memory.

    Args:
        path: Path to the video file.

    Returns:
        A VideoClip with all decoded frames.
    """
    reader = VideoReader(path)
    return VideoClip(frames=list(reader.frames()), fps=reader.fps)


def save_video(frames: list, fps: float, path: str) -> None:
    """Writes a list of frames to disk as an mp4.

    Args:
        frames: Non-empty list of BGR frames of equal size.
        fps: Output frame rate.
        path: Output video path.
    """
    h, w = frames[0].shape[:2]
    with VideoWriter(path, fps, w, h) as writer:
        for frame in frames:
            writer.write(frame)
