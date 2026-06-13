"""Player detection and tracking with YOLO26 + built-in tracker.

Player selection projects person detections to court coordinates and chooses
one candidate per court half. Track IDs are used for short-term continuity but
are allowed to change, which handles occlusions and players leaving the frame.
"""

from collections.abc import Sequence

import numpy as np
from ultralytics import YOLO

from ..geometry import (COURT_LENGTH, COURT_WIDTH_DOUBLES, NET_Y,
                        SERVICE_FROM_NET, CourtReference)
from ..progress import Progress

PERSON_CLASS = 0


def foot_point(bbox: np.ndarray) -> np.ndarray:
    """Returns where a player touches the court.

    Args:
        bbox: (x1, y1, x2, y2, ...) box; extra trailing values (e.g.
            confidence) are ignored.

    Returns:
        (x, y) bottom-center of the box, in pixels.
    """
    x1, y1, x2, y2 = bbox[:4]
    return np.array([(x1 + x2) / 2.0, y2])


class PlayerTracker:
    """Person detection and tracking with pretrained YOLO26."""

    def __init__(self, model_name: str = "models/yolo26x.pt",
                 conf: float = 0.3, imgsz: int = 1280):
        """Loads the model.

        Args:
            model_name: Path to pretrained YOLO26 detection weights.
            conf: Detection confidence threshold.
            imgsz: Inference resolution (longest side). The default 640
                downsamples a 1080p broadcast so much that the far player
                -- small and low-contrast, especially on clay -- is missed;
                1280 recovers it at the cost of slower inference.
        """
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz

    def track_frames(self, frames, total: int | None = None,
                     courts=None, court_crops: bool = False) -> list:
        """Runs detection + tracking on every frame.

        Args:
            frames: Any iterable of BGR frames (list or generator).
            total: Expected frame count, used for terminal progress.
            courts: Optional per-frame court references. Required when
                ``court_crops`` is enabled.
            court_crops: Run fallback inference on a court half when the
                full-frame detector finds no player near that baseline.

        Returns:
            Per frame, a dict {track_id: (5,) float array
            (x1, y1, x2, y2, conf)}.
        """
        if court_crops and courts is None:
            raise ValueError("courts are required for court-crop inference")
        detections = []
        progress = Progress(
            "player detection (court crops)" if court_crops
            else "player tracking",
            total,
        )
        for i, frame in enumerate(frames):
            if court_crops:
                result = self.model.predict(
                    frame, classes=[PERSON_CLASS], conf=self.conf,
                    imgsz=self.imgsz, verbose=False)[0]
                frame_dets = self._prediction_boxes(result)
                frame_court = courts[i]
                if frame_court is not None:
                    for player in self._missing_baseline_halves(
                            frame_dets, frame_court):
                        crop, offset = self._court_half_crop(
                            frame, frame_court, player)
                        if crop.size == 0:
                            continue
                        crop_result = self.model.predict(
                            crop, classes=[PERSON_CLASS],
                            conf=max(0.15, self.conf / 2.0),
                            imgsz=min(self.imgsz, 960),
                            verbose=False,
                        )[0]
                        recovered = self._prediction_boxes(
                            crop_result, offset=offset,
                            first_id=player * 1000)
                        frame_dets.update(recovered)
            else:
                result = self.model.track(
                    frame, persist=True, classes=[PERSON_CLASS],
                    conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
                frame_dets = {}
                if result.boxes is not None and result.boxes.id is not None:
                    ids = result.boxes.id.int().tolist()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    for tid, box, c in zip(ids, boxes, confs):
                        frame_dets[tid] = np.append(box, c)
            detections.append(frame_dets)
            progress.update(i + 1)
        progress.close(len(detections))
        return detections

    @staticmethod
    def _prediction_boxes(result, offset=(0, 0),
                          first_id: int = 0) -> dict:
        """Converts a prediction result to the tracker-compatible dictionary."""
        detections = {}
        if result.boxes is None:
            return detections
        ox, oy = offset
        for index, (box, confidence) in enumerate(zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy())):
            box = box.astype(np.float64)
            box[[0, 2]] += ox
            box[[1, 3]] += oy
            detections[first_id + index] = np.append(box, confidence)
        return detections

    @staticmethod
    def _missing_baseline_halves(detections: dict,
                                 court: CourtReference) -> list:
        """Returns player halves with no full-frame baseline candidate."""
        found = {1: False, 2: False}
        baseline_zone = SERVICE_FROM_NET + 2.5
        for bbox in detections.values():
            x, y = court.to_court(foot_point(bbox))[0]
            if not (-1.25 <= x <= COURT_WIDTH_DOUBLES + 1.25):
                continue
            player = 1 if y >= NET_Y else 2
            baseline_y = COURT_LENGTH if player == 1 else 0.0
            if abs(y - baseline_y) <= baseline_zone:
                found[player] = True
        return [player for player in (1, 2) if not found[player]]

    @staticmethod
    def _court_half_crop(frame: np.ndarray, court: CourtReference,
                         player: int):
        """Crops an expanded near/far court half for fallback detection."""
        y0, y1 = ((NET_Y, COURT_LENGTH) if player == 1
                  else (0.0, NET_Y))
        polygon = court.to_image(np.array([
            [0.0, y0],
            [COURT_WIDTH_DOUBLES, y0],
            [COURT_WIDTH_DOUBLES, y1],
            [0.0, y1],
        ]))
        x0, top = np.floor(polygon.min(axis=0)).astype(int)
        x1, bottom = np.ceil(polygon.max(axis=0)).astype(int)
        width = x1 - x0
        height = bottom - top
        x_margin = int(0.18 * width)
        top_margin = int(0.20 * height)
        bottom_margin = int((0.65 if player == 1 else 0.35) * height)
        x0 = max(0, x0 - x_margin)
        x1 = min(frame.shape[1], x1 + x_margin)
        top = max(0, top - top_margin)
        bottom = min(frame.shape[0], bottom + bottom_margin)
        return frame[top:bottom, x0:x1], (x0, top)

    @staticmethod
    def select_players(
        detections: list,
        court: CourtReference | Sequence[CourtReference],
        max_gap: int = 15,
    ) -> list:
        """Selects one player per court half and fills brief detection gaps.

        Broadcast trackers frequently assign a new ID after a player leaves the
        frame edge or is briefly occluded. Fixing two IDs for a whole rally then
        drops otherwise valid detections. Tennis gives a stronger identity cue:
        the players occupy opposite court halves. This method therefore assigns
        detections per frame by projected court position, preserving an existing
        track ID when possible and accepting a new ID near the corresponding
        baseline. Short bounded gaps are linearly interpolated.

        Args:
            detections: Per-frame dicts as returned by track_frames().
            court: One court reference for a static camera, or one reference
                per detection frame when the broadcast camera pans or zooms.
            max_gap: Maximum missing run, in frames, interpolated between two
                detections of the same player.

        Returns:
            Per frame, a dict {1: bbox, 2: bbox} with player 1 in the
            bottom half (near the camera).

        Raises:
            ValueError: If either court half has no player detection.
        """
        courts = ([court] * len(detections)
                  if isinstance(court, CourtReference) else list(court))
        if len(courts) != len(detections):
            raise ValueError("court sequence must match detections length")
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative")

        selected = [{} for _ in detections]
        state = {
            1: {"track_id": None, "position": None, "misses": max_gap + 1},
            2: {"track_id": None, "position": None, "misses": max_gap + 1},
        }
        baseline_zone = SERVICE_FROM_NET + 2.5

        for frame_index, (frame_dets, frame_court) in enumerate(
                zip(detections, courts)):
            candidates = {1: [], 2: []}
            for tid, bbox in frame_dets.items():
                pos = frame_court.to_court(foot_point(bbox))[0]
                x, y = pos
                if not (-1.25 <= x <= COURT_WIDTH_DOUBLES + 1.25
                        and -2.0 <= y <= COURT_LENGTH + 2.0):
                    continue
                player = 1 if y >= NET_Y else 2
                baseline_y = COURT_LENGTH if player == 1 else 0.0
                previous = state[player]
                same_track = (tid == previous["track_id"]
                              and previous["misses"] <= max_gap)
                spatially_continuous = False
                if (previous["position"] is not None
                        and previous["misses"] <= max_gap):
                    max_move = 2.0 + 0.35 * previous["misses"]
                    spatially_continuous = (
                        np.linalg.norm(pos - previous["position"]) <= max_move)
                continuing = same_track or spatially_continuous
                if abs(y - baseline_y) > baseline_zone and not continuing:
                    continue

                confidence = (float(bbox[4]) if len(bbox) > 4
                              and np.isfinite(bbox[4]) else 0.0)
                cost = abs(y - baseline_y) - 2.0 * confidence
                if same_track:
                    cost -= 20.0
                elif spatially_continuous:
                    cost -= 10.0
                if (previous["position"] is not None
                        and previous["misses"] <= max_gap):
                    cost += 0.25 * np.linalg.norm(
                        pos - previous["position"])
                candidates[player].append((cost, tid, bbox, pos))

            for player in (1, 2):
                if candidates[player]:
                    _, tid, bbox, pos = min(
                        candidates[player], key=lambda candidate: candidate[0])
                    selected[frame_index][player] = np.asarray(
                        bbox, dtype=np.float64)
                    state[player] = {
                        "track_id": tid,
                        "position": pos,
                        "misses": 0,
                    }
                else:
                    state[player]["misses"] += 1

        missing = [player for player in (1, 2)
                   if not any(player in frame for frame in selected)]
        if missing:
            raise ValueError(
                f"no detections for player half/halves {missing}")

        PlayerTracker._interpolate_player_gaps(selected, max_gap)
        return selected

    @staticmethod
    def _interpolate_player_gaps(frames: list, max_gap: int) -> None:
        """Interpolates bounding boxes across short, bounded missing runs."""
        if max_gap == 0:
            return
        for player in (1, 2):
            present = [i for i, frame in enumerate(frames) if player in frame]
            for left, right in zip(present, present[1:]):
                gap = right - left - 1
                if gap <= 0 or gap > max_gap:
                    continue
                box_left = frames[left][player]
                box_right = frames[right][player]
                for offset in range(1, gap + 1):
                    alpha = offset / (gap + 1)
                    box = np.full(5, np.nan, dtype=np.float64)
                    box[:4] = ((1.0 - alpha) * box_left[:4]
                               + alpha * box_right[:4])
                    frames[left + offset][player] = box
