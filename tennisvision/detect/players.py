"""Player detection and tracking with YOLO26 + built-in tracker.

Player selection strategy (differs from heuristics based on distance to
court keypoints): every person track is projected to court coordinates via
the homography, and the two track IDs that spend the largest fraction of
frames standing on the court are kept as the players. This is robust to
ball kids, line judges and spectators, who stand outside the playable area.
"""

from collections import defaultdict

import numpy as np
from ultralytics import YOLO

from ..geometry import CourtReference

PERSON_CLASS = 0


def foot_point(bbox: np.ndarray) -> np.ndarray:
    """Bottom-center of a bbox (x1,y1,x2,y2): where the player touches court."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, y2])


class PlayerTracker:
    def __init__(self, model_name: str = "yolo26x.pt", conf: float = 0.3):
        self.model = YOLO(model_name)
        self.conf = conf

    def track_frames(self, frames) -> list:
        """Returns, per frame, a dict {track_id: bbox (4,) float}.

        Accepts any iterable of frames (list or generator).
        """
        detections = []
        for frame in frames:
            result = self.model.track(
                frame, persist=True, classes=[PERSON_CLASS],
                conf=self.conf, verbose=False)[0]
            frame_dets = {}
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().tolist()
                boxes = result.boxes.xyxy.cpu().numpy()
                for tid, box in zip(ids, boxes):
                    frame_dets[tid] = box
            detections.append(frame_dets)
        return detections

    @staticmethod
    def select_players(detections: list, court: CourtReference) -> list:
        """Keep the two track IDs most consistently inside the court."""
        on_court = defaultdict(int)
        seen = defaultdict(int)
        for frame_dets in detections:
            for tid, bbox in frame_dets.items():
                seen[tid] += 1
                pos = court.to_court(foot_point(bbox))[0]
                if court.contains(pos):
                    on_court[tid] += 1
        ranked = sorted(seen, key=lambda t: on_court[t], reverse=True)
        players = set(ranked[:2])
        if len(players) < 2:
            raise ValueError("could not identify two players on court")

        # Relabel so that player 1 is the one in the bottom half (near camera)
        # at the first frame where both appear.
        id_map = None
        for frame_dets in detections:
            present = [t for t in players if t in frame_dets]
            if len(present) == 2:
                ys = {t: court.to_court(foot_point(frame_dets[t]))[0][1]
                      for t in present}
                bottom = max(ys, key=ys.get)
                top = min(ys, key=ys.get)
                id_map = {bottom: 1, top: 2}
                break
        if id_map is None:
            raise ValueError("players never visible in the same frame")

        return [
            {id_map[t]: bbox for t, bbox in frame_dets.items() if t in id_map}
            for frame_dets in detections
        ]
