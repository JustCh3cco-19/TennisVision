"""Rendering: annotated frames, top-down minimap, stats panel."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import (COURT_LINES, COURT_LENGTH, COURT_MODEL_POINTS,
                       COURT_WIDTH_DOUBLES, NET_Y)

PLAYER_COLORS = {1: (60, 180, 255), 2: (255, 120, 60)}  # BGR
BALL_COLOR = (0, 255, 255)


def draw_players(frame, player_boxes):
    """Draws player bounding boxes with id and confidence labels.

    Args:
        frame: BGR frame, modified in place.
        player_boxes: Dict {pid: bbox}; bbox is (x1, y1, x2, y2) with an
            optional 5th confidence element.

    Returns:
        The annotated frame.
    """
    for pid, bbox in player_boxes.items():
        x1, y1, x2, y2 = map(int, bbox[:4])
        color = PLAYER_COLORS.get(pid, (200, 200, 200))
        label = f"P{pid}"
        if len(bbox) > 4 and np.isfinite(bbox[4]):
            label += f" {bbox[4] * 100:.0f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


BALL_BOX_HALF = 22  # px; the smoothed track has no bbox size, fixed box


def draw_ball(frame, center, conf=None):
    """Draws the ball marker with an optional confidence label.

    Args:
        frame: BGR frame, modified in place.
        center: Smoothed (x, y) ball position in pixels; NaN skips drawing.
        conf: Detector confidence of the raw detection at this frame;
            NaN/None on interpolated frames omits the percentage.

    Returns:
        The annotated frame.
    """
    if np.isfinite(center).all():
        x, y = map(int, center)
        label = "ball"
        if conf is not None and np.isfinite(conf):
            label += f" {conf * 100:.0f}%"
        cv2.rectangle(frame, (x - BALL_BOX_HALF, y - BALL_BOX_HALF),
                      (x + BALL_BOX_HALF, y + BALL_BOX_HALF), BALL_COLOR, 2)
        cv2.putText(frame, label, (x - BALL_BOX_HALF, y - BALL_BOX_HALF - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BALL_COLOR, 2)
    return frame


def draw_ball_trail(frame, ball_px, frame_idx, tail: int = 30):
    """Draws the fitted ball trajectory as a fading trail.

    The smoothed track is the piecewise parabolic fit evaluated per
    frame, so the trail visualizes the fitted parabola itself. NaN
    frames (no segment coverage) naturally break the trail.

    Args:
        frame: BGR frame, modified in place.
        ball_px: (N, 2) smoothed ball positions in pixels.
        frame_idx: Current frame index; the trail covers the previous
            ``tail`` frames up to it.
        tail: Trail length in frames.

    Returns:
        The annotated frame.
    """
    start = max(0, frame_idx - tail)
    pts = ball_px[start:frame_idx + 1]
    for k in range(1, len(pts)):
        if not (np.isfinite(pts[k - 1]).all() and np.isfinite(pts[k]).all()):
            continue
        age = (k - 1) / max(1, len(pts) - 1)          # 0 = oldest, 1 = newest
        color = tuple(int(c * (0.25 + 0.75 * age)) for c in BALL_COLOR)
        thickness = 1 + int(round(2 * age))
        cv2.line(frame, tuple(np.int32(pts[k - 1])),
                 tuple(np.int32(pts[k])), color, thickness, cv2.LINE_AA)
    return frame


def draw_court_keypoints(frame, keypoints_px):
    """Draws detected court keypoints with their indices.

    Args:
        frame: BGR frame, modified in place.
        keypoints_px: (14, 2) keypoints in pixels, NaN entries skipped.

    Returns:
        The annotated frame.
    """
    for i, (x, y) in enumerate(keypoints_px):
        if np.isfinite((x, y)).all():
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return frame


RANSAC_INLIER_COLOR = (0, 255, 0)   # BGR, green
RANSAC_OUTLIER_COLOR = (0, 0, 255)  # BGR, red


def draw_ransac_keypoints(frame, court, legend: bool = True):
    """Draws the homography keypoints colored by RANSAC inlier status.

    Visualizes which detected court keypoints RANSAC kept (green inliers,
    drawn as filled circles) versus rejected (red outliers, drawn as
    hollow circles with a cross) when fitting the homography. This makes
    the robustness of the RANSAC fit directly visible: outliers are the
    mis-detected keypoints the homography correctly ignored.

    Args:
        frame: BGR frame, modified in place.
        court: A ``CourtReference`` carrying ``keypoints_px`` (N, 2) and the
            ``inliers`` (N,) boolean mask. None skips drawing.
        legend: If True, draws a small color legend in the top-left corner.

    Returns:
        The annotated frame.
    """
    if court is None:
        return frame
    for i, (p, is_in) in enumerate(zip(court.keypoints_px, court.inliers)):
        if not np.isfinite(p).all():
            continue
        x, y = int(p[0]), int(p[1])
        if is_in:
            cv2.circle(frame, (x, y), 5, RANSAC_INLIER_COLOR, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 6, RANSAC_OUTLIER_COLOR, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (x, y), RANSAC_OUTLIER_COLOR,
                           cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
        cv2.putText(frame, str(i), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    RANSAC_INLIER_COLOR if is_in else RANSAC_OUTLIER_COLOR, 1)
    if legend:
        n_in = int(np.count_nonzero(court.inliers))
        n_out = int(np.count_nonzero(
            np.isfinite(court.keypoints_px).all(axis=1) & ~court.inliers))
        _draw_ransac_legend(frame, n_in, n_out)
    return frame


def _draw_ransac_legend(frame, n_in, n_out, anchor=(20, 20)):
    """Draws the RANSAC inlier/outlier legend as a translucent card.

    Mirrors the look of the match-stats panel (rounded translucent card,
    TrueType font) so the counts stay readable over any background,
    instead of the bare colored text that washes out on light frames.

    Args:
        frame: BGR frame, modified in place.
        n_in: Number of inlier keypoints.
        n_out: Number of detected-but-rejected (outlier) keypoints.
        anchor: (x_left, y_top) of the card, in pixels.
    """
    pad, row_h, head_h = 12, 24, 24
    w, h = 210, pad * 2 + head_h + row_h * 2
    card = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=10, fill=_BG,
                        outline=(255, 255, 255, 40), width=1)
    gb, gg, gr = RANSAC_INLIER_COLOR        # stored BGR -> RGBA for PIL
    ob, og, orr = RANSAC_OUTLIER_COLOR
    green, red = (gr, gg, gb, 255), (orr, og, ob, 255)
    y = pad
    d.text((pad, y), "RANSAC", font=_FONT_TITLE, fill=_BRIGHT)
    y += head_h
    # inlier row: filled dot, matching the on-court inlier marker
    d.ellipse([pad, y + 4, pad + 11, y + 15], fill=green)
    d.text((pad + 22, y), f"Inlier ({n_in})", font=_FONT_VALUE, fill=_BRIGHT)
    y += row_h
    # outlier row: hollow circle + cross, matching the on-court outlier marker
    cx, cy = pad + 6, y + 9
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=red, width=2)
    d.line([cx - 4, cy - 4, cx + 4, cy + 4], fill=red, width=2)
    d.line([cx - 4, cy + 4, cx + 4, cy - 4], fill=red, width=2)
    d.text((pad + 22, y), f"Outlier ({n_out})", font=_FONT_VALUE, fill=_BRIGHT)
    _composite_card(frame, card, *anchor)
    return frame


def _composite_card(frame, card, x0, y0):
    """Alpha-composites an RGBA PIL card onto a BGR frame at (x0, y0).

    Args:
        frame: BGR frame, modified in place.
        card: RGBA PIL image to overlay.
        x0, y0: Top-left placement of the card, in pixels.

    Returns:
        The frame with the card composited.
    """
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    cw, ch = min(card.width, fw - x0), min(card.height, fh - y0)
    roi = cv2.cvtColor(frame[y0:y0 + ch, x0:x0 + cw], cv2.COLOR_BGR2RGB)
    out = Image.alpha_composite(Image.fromarray(roi).convert("RGBA"),
                                card.crop((0, 0, cw, ch)))
    frame[y0:y0 + ch, x0:x0 + cw] = cv2.cvtColor(
        np.asarray(out.convert("RGB")), cv2.COLOR_RGB2BGR)
    return frame


COURT_OVERLAY_COLOR = (0, 255, 0)  # BGR


def draw_court_overlay(frame, court, color=COURT_OVERLAY_COLOR,
                       thickness: int = 2, draw_vertices: bool = True):
    """Overlays the court model on the frame via the fitted homography.

    Projects the canonical metric court (``COURT_MODEL_POINTS`` /
    ``COURT_LINES``) back into image pixels with ``court.to_image`` and
    draws the line skeleton. When the homography is correct the projected
    lines coincide with the real court lines in the broadcast frame, which
    makes the RANSAC-fitted homography directly verifiable by eye.

    Args:
        frame: BGR frame, modified in place.
        court: A ``CourtReference`` (its ``to_image`` maps court meters to
            image pixels). None skips drawing (e.g. frames where the
            homography could not be estimated).
        color: BGR line color.
        thickness: Line thickness in pixels.
        draw_vertices: If True, also marks the projected model vertices.

    Returns:
        The annotated frame.
    """
    if court is None:
        return frame
    pts_px = court.to_image(COURT_MODEL_POINTS)            # (14, 2) in pixels
    for a, b in COURT_LINES:
        pa, pb = pts_px[a], pts_px[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(frame, tuple(np.int32(pa)), tuple(np.int32(pb)),
                     color, thickness, cv2.LINE_AA)
    if draw_vertices:
        for p in pts_px:
            if np.isfinite(p).all():
                cv2.circle(frame, tuple(np.int32(p)), 3, color, -1,
                           cv2.LINE_AA)
    # net line: midpoints of the two doubles sidelines (y = NET_Y in meters)
    net_px = court.to_image(np.array([[0.0, NET_Y],
                                      [COURT_WIDTH_DOUBLES, NET_Y]]))
    if np.isfinite(net_px).all():
        cv2.line(frame, tuple(np.int32(net_px[0])), tuple(np.int32(net_px[1])),
                 color, thickness, cv2.LINE_AA)
    return frame


class Minimap:
    """Top-down court rendered from the metric model.

    Attributes:
        w: Canvas width in pixels.
        h: Canvas height in pixels.
        base: Pre-rendered court background.
    """

    def __init__(self, scale: float = 8.0, pad: int = 20):
        """Pre-renders the court.

        Args:
            scale: Pixels per meter.
            pad: Canvas padding around the court, in pixels.
        """
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
                     self._to_px(COURT_MODEL_POINTS[b]),
                     (255, 255, 255), 2, cv2.LINE_AA)
        net_y = self._to_px((0, NET_Y))[1]
        cv2.line(img, (self.pad, net_y), (self.w - self.pad, net_y),
                 (50, 50, 50), 2, cv2.LINE_AA)
        return img

    def render(self, players_m: dict, ball_m=None, bounces_m=()):
        """Renders one minimap frame.

        Args:
            players_m: Dict {pid: (x, y)} player positions in meters.
            ball_m: Optional (x, y) ball position in meters.
            bounces_m: Recent bounce locations in meters, marked with
                small rings.

        Returns:
            BGR minimap image.
        """
        img = self.base.copy()
        for pos in bounces_m:
            if np.isfinite(pos).all():
                cv2.circle(img, self._to_px(pos), 4, (0, 200, 255), 1,
                           cv2.LINE_AA)
        for pid, pos in players_m.items():
            if np.isfinite(pos).all():
                cv2.circle(img, self._to_px(pos), 7,
                           PLAYER_COLORS.get(pid, (200, 200, 200)), -1,
                           cv2.LINE_AA)
        if ball_m is not None and np.isfinite(ball_m).all():
            cv2.circle(img, self._to_px(ball_m), 5, BALL_COLOR, -1,
                       cv2.LINE_AA)
        return img

    def paste(self, frame, minimap, margin: int = 20, y0: int = None):
        """Composites the minimap onto the right side of a frame.

        Args:
            frame: BGR frame, modified in place.
            minimap: Image returned by render().
            margin: Distance from the frame's right edge, pixels; also
                the top distance when ``y0`` is not given.
            y0: Optional top edge of the minimap, pixels.

        Returns:
            The frame with the minimap composited.
        """
        h, w = minimap.shape[:2]
        fh, fw = frame.shape[:2]
        if y0 is None:
            y0 = margin
        x0 = fw - w - margin
        roi = frame[y0:y0 + h, x0:x0 + w]
        frame[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(
            roi, 0.1, minimap, 0.9, 0)
        cv2.rectangle(frame, (x0, y0), (x0 + w - 1, y0 + h - 1),
                      (240, 240, 240), 1, cv2.LINE_AA)
        return frame


_FONT_DIRS = ["/usr/share/fonts/truetype/ubuntu",
              "/usr/share/fonts/truetype/dejavu"]


def _load_font(names, size):
    """Loads the first available TrueType font from _FONT_DIRS.

    Args:
        names: Font file names to try, in order of preference.
        size: Point size.

    Returns:
        An ImageFont; PIL's bitmap default if none of the files exist.
    """
    for d in _FONT_DIRS:
        for name in names:
            p = Path(d) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


_FONT_TITLE = _load_font(["Ubuntu-B.ttf", "DejaVuSans-Bold.ttf"], 17)
_FONT_LABEL = _load_font(["Ubuntu-L.ttf", "DejaVuSans.ttf"], 15)
_FONT_VALUE = _load_font(["Ubuntu-M.ttf", "Ubuntu-B.ttf",
                          "DejaVuSans-Bold.ttf"], 15)

PANEL_W = 260
_BG = (16, 18, 28, 215)
_DIM = (165, 170, 185, 255)
_BRIGHT = (245, 246, 250, 255)


def stats_panel_height() -> int:
    """Returns the rendered height of the stats card, in pixels."""
    pad, row_h, head_h, sub_h = 14, 22, 26, 20
    n_rows, n_subs = 5, 2
    return (pad + 24
            + 2 * (head_h + n_subs * sub_h + n_rows * row_h + 8)
            + pad - 8)


def draw_stats_panel(frame, stats, frame_idx, anchor=None):
    """Draws the translucent match-stats card.

    Per player, metrics are split into a "Ball" subgroup (shot count,
    last shot speed and type, average shot speed) and a "Player"
    subgroup (movement speed), all up to the current frame.

    Args:
        frame: BGR frame, modified in place.
        stats: MatchStats with shots and per-frame player speeds.
        frame_idx: Current frame index.
        anchor: (x_right, y_top) of the card; defaults to top-right.

    Returns:
        The frame with the card composited.
    """
    pad, row_h, head_h, sub_h = 14, 22, 26, 20
    h = stats_panel_height()
    x1, y0 = anchor if anchor else (frame.shape[1] - 20, 20)
    x0 = x1 - PANEL_W

    card = Image.new("RGBA", (PANEL_W, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([0, 0, PANEL_W - 1, h - 1], radius=12, fill=_BG,
                        outline=(255, 255, 255, 40), width=1)
    y = pad
    d.text((pad, y), "MATCH STATS", font=_FONT_TITLE, fill=_BRIGHT)
    y += 24
    for pid in (1, 2):
        b, g, r = PLAYER_COLORS[pid]
        accent = (r, g, b, 255)
        d.ellipse([pad, y + 6, pad + 10, y + 16], fill=accent)
        d.text((pad + 18, y + 2), f"PLAYER {pid}",
               font=_FONT_VALUE, fill=accent)
        y += head_h
        groups = [
            ("BALL", [
                ("Shots", f"{stats.shot_count(pid, frame_idx)}"),
                ("Last shot",
                 f"{stats.last_shot_speed(pid, frame_idx):.1f} km/h"),
                ("Last type", stats.last_shot_type(pid, frame_idx)),
                ("Avg shot",
                 f"{stats.avg_shot_speed(pid, frame_idx):.1f} km/h"),
            ]),
            ("PLAYER", [
                ("Speed",
                 f"{stats.movement_speed(pid, frame_idx):.1f} km/h"),
            ]),
        ]
        for sub, rows in groups:
            d.text((pad + 18, y + 2), sub, font=_FONT_LABEL, fill=_DIM)
            d.line([pad + 18 + 6 * len(sub) + 10, y + 11,
                    PANEL_W - pad, y + 11], fill=(255, 255, 255, 30))
            y += sub_h
            for label, value in rows:
                d.text((pad + 30, y), label, font=_FONT_LABEL, fill=_DIM)
                d.text((PANEL_W - pad, y), value, font=_FONT_VALUE,
                       fill=_BRIGHT, anchor="ra")
                y += row_h
        y += 8

    # alpha-composite the card onto the BGR frame
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    cw, ch = min(PANEL_W, fw - x0), min(h, fh - y0)
    roi = cv2.cvtColor(frame[y0:y0 + ch, x0:x0 + cw], cv2.COLOR_BGR2RGB)
    out = Image.alpha_composite(Image.fromarray(roi).convert("RGBA"),
                                card.crop((0, 0, cw, ch)))
    frame[y0:y0 + ch, x0:x0 + cw] = cv2.cvtColor(
        np.asarray(out.convert("RGB")), cv2.COLOR_RGB2BGR)
    return frame
