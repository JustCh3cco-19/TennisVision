"""Rendering: annotated frames, top-down minimap, stats panel."""

import cv2
import numpy as np

from .geometry import (COURT_LINES, COURT_LENGTH, COURT_MODEL_POINTS,
                       COURT_WIDTH_DOUBLES)

PLAYER_COLORS = {1: (60, 180, 255), 2: (255, 120, 60)}  # BGR
BALL_COLOR = (0, 255, 255)


def draw_players(frame, player_boxes):
    for pid, bbox in player_boxes.items():
        x1, y1, x2, y2 = map(int, bbox)
        color = PLAYER_COLORS.get(pid, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"P{pid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


BALL_BOX_HALF = 22  # px; the smoothed track has no bbox size, fixed box


def draw_ball(frame, center):
    if np.isfinite(center).all():
        x, y = map(int, center)
        cv2.rectangle(frame, (x - BALL_BOX_HALF, y - BALL_BOX_HALF),
                      (x + BALL_BOX_HALF, y + BALL_BOX_HALF), BALL_COLOR, 2)
        cv2.putText(frame, "ball", (x - BALL_BOX_HALF, y - BALL_BOX_HALF - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BALL_COLOR, 2)
    return frame


def draw_shot_speed(frame, stats, frame_idx, player_boxes,
                    hold_frames: int = 36):
    """For ~hold_frames after a hit, write the shot speed above the bbox of
    the player who hit it (falls back to nothing if the box is missing)."""
    recent = [s for s in stats.shots
              if s.frame <= frame_idx < s.frame + hold_frames]
    if not recent:
        return frame
    s = recent[-1]
    bbox = player_boxes.get(s.player)
    if bbox is None:
        return frame
    x1, y1, x2, _ = map(int, bbox)
    text = f"{s.ball_speed_kmh:.0f} km/h"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    # above the bbox, next to the P{id} label; clamp inside the frame
    x = max(0, min((x1 + x2 - tw) // 2, frame.shape[1] - tw))
    y = max(th + 10, y1 - 14)
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 6),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale,
                PLAYER_COLORS.get(s.player, (255, 255, 255)), thick)
    return frame


def draw_court_keypoints(frame, keypoints_px):
    for i, (x, y) in enumerate(keypoints_px):
        if np.isfinite((x, y)).all():
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return frame


class Minimap:
    """Top-down court rendered from the metric model."""

    def __init__(self, scale: float = 8.0, pad: int = 20):
        self.scale = scale
        self.pad = pad
        self.w = int(COURT_WIDTH_DOUBLES * scale) + 2 * pad
        self.h = int(COURT_LENGTH * scale) + 2 * pad
        self.base = self._render_court()

    def _to_px(self, pt_m):
        return (int(pt_m[0] * self.scale) + self.pad,
                int(pt_m[1] * self.scale) + self.pad)

    def _render_court(self):
        img = np.full((self.h, self.w, 3), (90, 140, 60), np.uint8)
        for a, b in COURT_LINES:
            cv2.line(img, self._to_px(COURT_MODEL_POINTS[a]),
                     self._to_px(COURT_MODEL_POINTS[b]), (255, 255, 255), 2)
        net_y = self._to_px((0, COURT_LENGTH / 2))[1]
        cv2.line(img, (self.pad, net_y), (self.w - self.pad, net_y),
                 (50, 50, 50), 2)
        return img

    def render(self, players_m: dict, ball_m=None):
        img = self.base.copy()
        for pid, pos in players_m.items():
            if np.isfinite(pos).all():
                cv2.circle(img, self._to_px(pos), 7,
                           PLAYER_COLORS.get(pid, (200, 200, 200)), -1)
        if ball_m is not None and np.isfinite(ball_m).all():
            cv2.circle(img, self._to_px(ball_m), 5, BALL_COLOR, -1)
        return img

    def paste(self, frame, minimap, margin: int = 20):
        h, w = minimap.shape[:2]
        fh, fw = frame.shape[:2]
        y0, x0 = margin, fw - w - margin
        roi = frame[y0:y0 + h, x0:x0 + w]
        frame[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(roi, 0.3, minimap, 0.7, 0)
        return frame


def draw_stats_panel(frame, stats, frame_idx):
    lines = [f"frame {frame_idx}"]
    for pid in (1, 2):
        lines.append(
            f"P{pid}  shots: {sum(1 for s in stats.shots if s.player == pid and s.frame <= frame_idx)}"
            f"  last: {stats.last_shot_speed(pid, frame_idx):5.1f} km/h")
    x, y = 20, frame.shape[0] - 90
    cv2.rectangle(frame, (x - 10, y - 25), (x + 380, y + 55), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame
