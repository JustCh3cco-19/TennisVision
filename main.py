"""Tennis match analysis pipeline (YOLO26-based).

Usage:
    python main.py --video input/match.mp4 \
        --ball-model models/ball_yolo26.pt \
        --court-model models/court_pose_yolo26.pt \
        --output output/annotated.mp4
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from tennisvision import (BallParabolicSmoother, CourtReference, compute_stats,
                         detect_bounces, detect_hits,
                         filter_ball_detections_to_court,
                         filter_static_ball_detections, VideoReader,
                         VideoWriter)
from tennisvision.detect import (BallDetector, CourtKeypointDetector,
                                PlayerTracker, foot_point, white_line_mask)
from tennisvision.geometry import COURT_MODEL_POINTS, smooth_court_keypoints
from tennisvision.progress import Progress
from tennisvision import viz


def parse_args():
    """Parses the command-line arguments.

    Returns:
        The populated argparse.Namespace.
    """
    p = argparse.ArgumentParser(description="Tennis video analysis")
    p.add_argument("--video", required=True)
    p.add_argument("--ball-model", required=True)
    p.add_argument("--ball-imgsz", type=int, default=1280,
                   help="Inference resolution for the small tennis ball; "
                        "1280 improves recall on grass/clay over the default "
                        "640 at additional cost (default: 1280)")
    p.add_argument("--court-model", required=True)
    p.add_argument("--player-model", default="models/yolo26x.pt")
    p.add_argument("--player-imgsz", type=int, default=1280,
                   help="Inference resolution for the player detector; "
                        "raise it (e.g. 1280) so the small, low-contrast "
                        "far player is not lost in downsampling, lower it "
                        "(640) for faster inference (default: 1280)")
    p.add_argument("--player-max-gap", type=int, default=15,
                   help="Maximum short player-detection gap interpolated "
                        "within a court-view segment (default: 15 frames)")
    p.add_argument("--player-court-crops", action="store_true",
                   help="Run fallback player inference on a missing court "
                        "half; improves grass/clay recall but is slower")
    p.add_argument("--output", default="output/annotated.mp4")
    p.add_argument("--cache", default=None,
                   help="directory for cached detections (skips inference)")
    p.add_argument("--show", action="store_true",
                   help="display annotated frames live while rendering (q to stop)")
    p.add_argument("--ransac-thresh", type=float, default=0.4,
                   help="RANSAC reprojection threshold for the court "
                        "homography, in meters (default: 0.4)")
    p.add_argument("--court-conf", type=float, default=0.5,
                   help="Per-keypoint confidence threshold for the court "
                        "model; lower it on clay/grass/natural light to "
                        "keep more (noisier) keypoints for RANSAC "
                        "(default: 0.5)")
    p.add_argument("--court-min-keypoints", type=int, default=8,
                   help="Minimum confident court keypoints required to accept "
                        "a frame as a court view (default: 8)")
    p.add_argument("--court-smooth-window", type=int, default=11,
                   help="Odd temporal median window for per-frame court "
                        "keypoints; tracks camera pans/zooms while suppressing "
                        "pose jitter (default: 11)")
    p.add_argument("--court-preprocess", action="store_true",
                   help="CLAHE-normalize each frame before court detection "
                        "(helps on clay/grass and natural light)")
    p.add_argument("--court-refine", action="store_true",
                   help="refine the court homography by snapping the model "
                        "lines onto the detected white lines (chamfer ICP); "
                        "corrects the overlay drift on clay/grass")
    return p.parse_args()


def cached(cache_dir, name, fn):
    """Runs fn() or loads its pickled result.

    Lets the analysis stages iterate quickly without re-running inference.

    Args:
        cache_dir: Cache directory, or None to disable caching.
        name: Cache key; the result is stored as ``<name>.pkl``.
        fn: Zero-argument callable producing the value to cache.

    Returns:
        The cached or freshly computed value.
    """
    if cache_dir is None:
        print("  cache disabled: running inference")
        return fn()
    path = Path(cache_dir) / f"{name}.pkl"
    if path.exists():
        print(f"  cache hit: {path}")
        return pickle.loads(path.read_bytes())
    print(f"  cache miss: {path}")
    result = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(result))
    print(f"  cache saved: {path}")
    return result


def visible_segments(mask) -> list:
    """Finds contiguous runs of True in a boolean mask.

    Camera cuts are the natural segment boundaries.

    Args:
        mask: 1-D boolean array.

    Returns:
        (start, end) pairs, end exclusive.
    """
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2], edges[1::2]))


def fit_court_sequence(kps_frames, court_visible, ransac_thresh,
                       smooth_window):
    """Fits a temporally stabilized homography for every visible frame."""
    print(f"  smoothing keypoints with a {smooth_window}-frame median")
    smoothed = smooth_court_keypoints(
        kps_frames, court_visible, window=smooth_window)
    courts = [None] * len(kps_frames)
    visible_idx = np.flatnonzero(court_visible)
    progress = Progress("homography fitting", len(visible_idx))
    for completed, i in enumerate(visible_idx, start=1):
        try:
            courts[i] = CourtReference.from_keypoints(
                smoothed[i], ransac_thresh=ransac_thresh)
        except ValueError:
            court_visible[i] = False
        progress.update(completed)
    progress.close(len(visible_idx))
    return courts


def refine_court_sequence(courts, court_visible, video, ransac_thresh,
                          stride: int = 25):
    """Line-refines sparse frames and interpolates the correction over time.

    Refining a single mask aggregated over the full video is invalid when the
    camera pans or zooms. Instead, this computes a small image-space correction
    on sparse anchor frames, then interpolates that correction independently
    inside each contiguous court-view segment.
    """
    refined_courts = list(courts)
    n_refined = 0
    segments = visible_segments(court_visible)
    anchor_groups = [
        np.unique(np.r_[np.arange(start, end, stride), end - 1])
        for start, end in segments
    ]
    progress = Progress(
        "court line refinement",
        sum(len(anchors) for anchors in anchor_groups),
    )
    completed = 0
    for (start, end), anchors in zip(segments, anchor_groups):
        residuals = []
        valid_anchors = []
        for i in anchors:
            base = courts[int(i)]
            if base is None:
                continue
            frame = video.frame_at(int(i))
            refined = base.refine_to_lines(white_line_mask(frame))
            residuals.append(
                refined.to_image(COURT_MODEL_POINTS)
                - base.to_image(COURT_MODEL_POINTS))
            valid_anchors.append(int(i))
            n_refined += refined is not base
            completed += 1
            progress.update(completed)
        if not valid_anchors:
            continue

        valid_anchors = np.asarray(valid_anchors)
        residuals = np.asarray(residuals)
        for i in range(start, end):
            base = courts[i]
            if base is None:
                continue
            right = int(np.searchsorted(valid_anchors, i))
            if right == 0:
                correction = residuals[0]
            elif right == len(valid_anchors):
                correction = residuals[-1]
            else:
                left = right - 1
                span = valid_anchors[right] - valid_anchors[left]
                alpha = (i - valid_anchors[left]) / span
                correction = ((1.0 - alpha) * residuals[left]
                              + alpha * residuals[right])
            adjusted = base.to_image(COURT_MODEL_POINTS) + correction
            fitted = CourtReference.from_keypoints(
                adjusted, ransac_thresh=ransac_thresh)
            refined_courts[i] = CourtReference(
                homography=fitted.homography,
                inverse=fitted.inverse,
                keypoints_px=base.keypoints_px,
                inliers=base.inliers,
            )
    progress.close(completed)
    return refined_courts, n_refined


def main():
    """Runs the full analysis pipeline and writes the annotated video."""
    args = parse_args()
    if not 4 <= args.court_min_keypoints <= len(COURT_MODEL_POINTS):
        raise SystemExit("--court-min-keypoints must be between 4 and 14")
    if args.court_smooth_window < 1 or args.court_smooth_window % 2 == 0:
        raise SystemExit("--court-smooth-window must be a positive odd integer")
    if args.player_max_gap < 0:
        raise SystemExit("--player-max-gap must be non-negative")

    # frames are streamed from disk on each pass: a full 1080p match does
    # not fit in RAM decoded.
    video = VideoReader(args.video)
    print(f"Input: {args.video}")
    print(f"  {video.n_frames} frames | {video.fps:.1f} fps | "
          f"{video.width}x{video.height}")

    # 1. Court keypoints also classify each frame as court / no-court.
    # A homography is fitted per frame. Broadcast cameras pan and zoom even
    # within a rally, so one median homography for the full clip drifts badly.
    # A short temporal median stabilizes detector jitter without freezing the
    # changing camera pose.
    print("\n[1/6] Court detection and homography")
    court_detector = CourtKeypointDetector(
        args.court_model, kpt_conf=args.court_conf,
        preprocess=args.court_preprocess)
    default_court_settings = (args.court_conf == 0.5
                              and not args.court_preprocess)
    court_cache_key = (
        "court_frames" if default_court_settings else
        f"court_frames_conf{args.court_conf:g}"
        f"_prep{int(args.court_preprocess)}")
    kps_frames = cached(args.cache, court_cache_key,
                        lambda: court_detector.detect_frames(
                            video.frames(), total=video.n_frames))
    valid_kpt_count = np.isfinite(kps_frames).all(axis=2).sum(axis=1)
    court_visible = valid_kpt_count >= args.court_min_keypoints
    # fill brief detection dropouts: a single missed frame is a detector
    # hiccup, not a camera cut, and must not split a rally in two segments
    for start, end in visible_segments(~court_visible):
        if end - start <= 5 and start > 0 and end < video.n_frames:
            court_visible[start:end] = True
    if not court_visible.any():
        raise SystemExit("no court detected in any frame")
    print(f"  court visible: {court_visible.sum()}/{video.n_frames} frames")
    courts = fit_court_sequence(
        kps_frames, court_visible, args.ransac_thresh,
        args.court_smooth_window)

    # 1b. (optional) snap sparse per-frame poses onto the real white lines and
    # interpolate the small correction. A full-video static mask cannot be
    # used because the broadcast camera moves.
    if args.court_refine:
        print("  refining projected court lines on sparse anchor frames")
        courts, n_refined = refine_court_sequence(
            courts, court_visible, video, args.ransac_thresh)
        print(f"  accepted refinements: {n_refined} anchor frames")
    else:
        print("  line refinement: disabled")

    # 2. players (detections on cutaway frames are meaningless: dropped).
    # Track IDs do not survive camera cuts (the tracker re-assigns fresh IDs
    # when the court view comes back), so P1/P2 are selected independently
    # within each contiguous court-visible segment.
    print("\n[2/6] Player detection and selection")
    tracker = PlayerTracker(args.player_model, imgsz=args.player_imgsz)
    # cache key v2: boxes now carry a 5th confidence column; the inference
    # resolution is part of the key so different --player-imgsz settings do
    # not silently reuse each other's boxes.
    player_cache_key = (
        f"players_v3_imgsz{args.player_imgsz}_crops1"
        if args.player_court_crops else
        f"players_v2_imgsz{args.player_imgsz}")
    raw_tracks = cached(args.cache, player_cache_key,
                        lambda: tracker.track_frames(
                            video.frames(), total=video.n_frames,
                            courts=courts,
                            court_crops=args.player_court_crops))
    raw_tracks = [fr if vis else {}
                  for fr, vis in zip(raw_tracks, court_visible)]
    player_boxes = [{} for _ in range(video.n_frames)]
    court_segments = visible_segments(court_visible)
    progress = Progress("player selection", len(court_segments))
    for segment_index, (start, end) in enumerate(court_segments, start=1):
        try:
            player_boxes[start:end] = tracker.select_players(
                raw_tracks[start:end], courts[start:end],
                max_gap=args.player_max_gap)
        except ValueError as e:
            print(f"\n  frames {start}-{end}: players not identified ({e})")
        progress.update(segment_index)
    progress.close(len(court_segments))
    print("  projecting player positions to court coordinates")
    players_court = [
        {pid: courts[i].to_court(foot_point(b))[0]
         for pid, b in fr.items()}
        for i, fr in enumerate(player_boxes)
    ]

    # 3. ball: detect -> piecewise parabolic fit -> project to meters
    print("\n[3/6] Ball detection and trajectory")
    ball_detector = BallDetector(args.ball_model, imgsz=args.ball_imgsz)
    # Resolution is part of the key because it materially changes recall.
    raw_ball = cached(args.cache, f"ball_v3_imgsz{args.ball_imgsz}",
                      lambda: ball_detector.detect_frames(
                          video.frames(), total=video.n_frames))
    raw_ball_count = int(np.isfinite(raw_ball[:, :2]).all(axis=1).sum())
    raw_ball = filter_ball_detections_to_court(
        raw_ball, courts, court_visible)
    court_filtered_count = int(
        np.isfinite(raw_ball[:, :2]).all(axis=1).sum())
    raw_ball = filter_static_ball_detections(raw_ball)
    filtered_ball_count = int(
        np.isfinite(raw_ball[:, :2]).all(axis=1).sum())
    print(f"  rejected off-court detections: "
          f"{raw_ball_count - court_filtered_count}")
    print(f"  rejected static detections: "
          f"{court_filtered_count - filtered_ball_count}")
    ball_conf = raw_ball[:, 2]
    print("  smoothing ball trajectory")
    ball_px = BallParabolicSmoother().smooth(raw_ball[:, :2])
    ball_px[~court_visible] = np.nan  # don't let the fit bridge camera cuts
    ball_court = np.full_like(ball_px, np.nan)
    valid = np.isfinite(ball_px).all(axis=1)
    valid_ball_idx = np.flatnonzero(valid)
    progress = Progress("ball projection", len(valid_ball_idx))
    for completed, i in enumerate(valid_ball_idx, start=1):
        ball_court[i] = courts[i].to_court(ball_px[i])[0]
        progress.update(completed)
    progress.close(len(valid_ball_idx))

    # 4. events + stats
    print("\n[4/6] Event detection")
    hit_events = detect_hits(
        ball_court, video.fps, ball_px, player_boxes,
        return_players=True)
    hits = [frame for frame, _ in hit_events]
    bounces = detect_bounces(ball_px, ball_court, hits, players_court,
                             video.fps)
    print(f"  detected bounces: {len(bounces)}")
    print("\n[5/6] Statistics")
    stats = compute_stats(
        hits, ball_court, players_court, video.fps,
        hit_players=dict(hit_events))
    print(f"  detected shots: {len(hits)}")
    for s in stats.shots:
        print(f"  frame {s.frame:5d}  P{s.player}  {s.shot_type:<12s}"
              f"ball {s.ball_speed_kmh:5.1f} km/h  "
              f"opponent {s.opponent_speed_kmh:4.1f} km/h")

    # 5. render: streamed, each annotated frame is written to disk immediately
    print("\n[6/6] Rendering annotated video")
    minimap = viz.Minimap()
    # vertically center the minimap + stats card block on the right edge
    block_h = minimap.h + 12 + viz.stats_panel_height()
    block_y = max(20, (video.height - block_h) // 2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    progress = Progress("video rendering", video.n_frames)
    with VideoWriter(args.output, video.fps, video.width, video.height) as writer:
        for i, frame in enumerate(video.frames()):
            viz.draw_players(frame, player_boxes[i])
            viz.draw_ball_trail(frame, ball_px, i)
            viz.draw_ball(frame, ball_px[i], ball_conf[i])
            if court_visible[i]:
                viz.draw_court_overlay(frame, courts[i])
                viz.draw_ransac_keypoints(frame, courts[i])
            recent = [ball_court[b] for b in bounces
                      if b <= i <= b + int(1.5 * video.fps)]
            mm = minimap.render(players_court[i], ball_court[i], recent)
            frame = minimap.paste(frame, mm, y0=block_y)
            viz.draw_stats_panel(frame, stats, i,
                                 anchor=(video.width - 20,
                                         block_y + minimap.h + 12))
            writer.write(frame)
            progress.update(i + 1)
            if args.show:
                cv2.imshow("TennisVision", frame)
                if cv2.waitKey(max(1, int(1000 / video.fps))) & 0xFF == ord("q"):
                    args.show = False
                    cv2.destroyAllWindows()
    progress.close(video.n_frames)

    cv2.destroyAllWindows()
    print(f"\nCompleted: {args.output}")


if __name__ == "__main__":
    main()
