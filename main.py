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
                         detect_bounces, detect_hits, VideoReader, VideoWriter)
from tennisvision.detect import (BallDetector, CourtKeypointDetector,
                                PlayerTracker, foot_point)
from tennisvision import viz


def parse_args():
    """Parses the command-line arguments.

    Returns:
        The populated argparse.Namespace.
    """
    p = argparse.ArgumentParser(description="Tennis video analysis")
    p.add_argument("--video", required=True)
    p.add_argument("--ball-model", required=True)
    p.add_argument("--court-model", required=True)
    p.add_argument("--player-model", default="models/yolo26x.pt")
    p.add_argument("--output", default="output/annotated.mp4")
    p.add_argument("--cache", default=None,
                   help="directory for cached detections (skips inference)")
    p.add_argument("--show", action="store_true",
                   help="display annotated frames live while rendering (q to stop)")
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
        return fn()
    path = Path(cache_dir) / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    result = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(result))
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


def main():
    """Runs the full analysis pipeline and writes the annotated video."""
    args = parse_args()
    # frames are streamed from disk on each pass: a full 1080p match does
    # not fit in RAM decoded.
    video = VideoReader(args.video)
    print(f"opened {video.n_frames} frames @ {video.fps:.1f} fps")

    # 1. court keypoints, per frame: broadcast footage cuts away from the
    # court view, so each frame is classified as court / no-court. The
    # homography itself is still estimated once (median over court frames):
    # the main camera is static, and a single stable fit beats per-frame
    # jitter for the metric projection.
    court_detector = CourtKeypointDetector(args.court_model)
    kps_frames = cached(args.cache, "court_frames",
                        lambda: court_detector.detect_frames(video.frames()))
    court_visible = np.isfinite(kps_frames).all(axis=2).sum(axis=1) >= 4
    # fill brief detection dropouts: a single missed frame is a detector
    # hiccup, not a camera cut, and must not split a rally in two segments
    for start, end in visible_segments(~court_visible):
        if end - start <= 5 and start > 0 and end < video.n_frames:
            court_visible[start:end] = True
    if not court_visible.any():
        raise SystemExit("no court detected in any frame")
    print(f"court visible in {court_visible.sum()}/{video.n_frames} frames")
    keypoints = np.nanmedian(kps_frames[court_visible], axis=0)
    court = CourtReference.from_keypoints(keypoints)

    # 2. players (detections on cutaway frames are meaningless: dropped).
    # Track IDs do not survive camera cuts (the tracker re-assigns fresh IDs
    # when the court view comes back), so P1/P2 are selected independently
    # within each contiguous court-visible segment.
    tracker = PlayerTracker(args.player_model)
    # cache key v2: boxes now carry a 5th confidence column
    raw_tracks = cached(args.cache, "players_v2",
                        lambda: tracker.track_frames(video.frames()))
    raw_tracks = [fr if vis else {}
                  for fr, vis in zip(raw_tracks, court_visible)]
    player_boxes = [{} for _ in range(video.n_frames)]
    for start, end in visible_segments(court_visible):
        try:
            player_boxes[start:end] = tracker.select_players(
                raw_tracks[start:end], court)
        except ValueError as e:
            print(f"frames {start}-{end}: players not identified ({e})")
    players_court = [
        {pid: court.to_court(foot_point(b))[0] for pid, b in fr.items()}
        for fr in player_boxes
    ]

    # 3. ball: detect -> piecewise parabolic fit -> project to meters
    ball_detector = BallDetector(args.ball_model)
    # cache key v2: detections now carry an (x, y, conf) third column
    raw_ball = cached(args.cache, "ball_v2",
                      lambda: ball_detector.detect_frames(video.frames()))
    raw_ball = raw_ball.copy()
    raw_ball[~court_visible] = np.nan  # ball "detections" on cutaway frames
    ball_conf = raw_ball[:, 2]
    ball_px = BallParabolicSmoother().smooth(raw_ball[:, :2])
    ball_px[~court_visible] = np.nan  # don't let the fit bridge camera cuts
    ball_court = np.full_like(ball_px, np.nan)
    valid = np.isfinite(ball_px).all(axis=1)
    if valid.any():
        ball_court[valid] = court.to_court(ball_px[valid])

    # 4. events + stats
    hits = detect_hits(ball_court, video.fps, ball_px, player_boxes)
    bounces = detect_bounces(ball_px, ball_court, hits, players_court,
                             video.fps)
    print(f"detected {len(bounces)} bounces")
    stats = compute_stats(hits, ball_court, players_court, video.fps)
    print(f"detected {len(hits)} shots")
    for s in stats.shots:
        print(f"  frame {s.frame:5d}  P{s.player}  {s.shot_type:<12s}"
              f"ball {s.ball_speed_kmh:5.1f} km/h  "
              f"opponent {s.opponent_speed_kmh:4.1f} km/h")

    # 5. render: streamed, each annotated frame is written to disk immediately
    minimap = viz.Minimap()
    # vertically center the minimap + stats card block on the right edge
    block_h = minimap.h + 12 + viz.stats_panel_height()
    block_y = max(20, (video.height - block_h) // 2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with VideoWriter(args.output, video.fps, video.width, video.height) as writer:
        for i, frame in enumerate(video.frames()):
            viz.draw_players(frame, player_boxes[i])
            viz.draw_ball_trail(frame, ball_px, i)
            viz.draw_ball(frame, ball_px[i], ball_conf[i])
            viz.draw_court_keypoints(frame, kps_frames[i])
            recent = [ball_court[b] for b in bounces
                      if b <= i <= b + int(1.5 * video.fps)]
            mm = minimap.render(players_court[i], ball_court[i], recent)
            frame = minimap.paste(frame, mm, y0=block_y)
            viz.draw_stats_panel(frame, stats, i,
                                 anchor=(video.width - 20,
                                         block_y + minimap.h + 12))
            writer.write(frame)
            if args.show:
                cv2.imshow("TennisVision", frame)
                if cv2.waitKey(max(1, int(1000 / video.fps))) & 0xFF == ord("q"):
                    args.show = False
                    cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
