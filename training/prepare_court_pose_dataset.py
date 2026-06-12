"""Convert a local tennis court keypoint dataset to Ultralytics pose format.

Input format: data.json with 14 keypoints per image, plus a local image folder.
Example JSON item: {"id": ..., "kps": [[x, y], ...]} for 1280x720 images.

IMPORTANT: the keypoint ordering of the source dataset must be remapped to
the canonical ordering in tennisvision.geometry.COURT_MODEL_POINTS. Adjust
SOURCE_TO_CANONICAL after visually checking a few annotated samples
(use --preview to dump one).

Output layout (Ultralytics pose):
    out/
      images/{train,val}/*.png
      labels/{train,val}/*.txt   # class cx cy w h  (x y vis) * 14, normalized
      court-pose.yaml
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

# index in source dataset -> index in canonical model (geometry.py).
# Identity by default: VERIFY against a preview image before training.
SOURCE_TO_CANONICAL = list(range(14))

YAML = """\
path: {root}
train: images/train
val: images/val
kpt_shape: [14, 3]
names:
  0: court
"""


def convert(kps, w, h):
    """Converts one annotation to an Ultralytics pose label line.

    Args:
        kps: List of 14 (x, y) keypoints in pixels, source ordering.
        w: Image width in pixels.
        h: Image height in pixels.

    Returns:
        The label line "class cx cy w h (x y vis)*14" with normalized
        coordinates, or None if fewer than 4 keypoints are visible.
    """
    pts = np.full((14, 3), 0.0)
    for src_i, (x, y) in enumerate(kps):
        dst_i = SOURCE_TO_CANONICAL[src_i]
        visible = 0 <= x < w and 0 <= y < h
        pts[dst_i] = (x / w, y / h, 2.0 if visible else 0.0)
    vis = pts[pts[:, 2] > 0]
    if len(vis) < 4:
        return None
    x0, y0 = vis[:, 0].min(), vis[:, 1].min()
    x1, y1 = vis[:, 0].max(), vis[:, 1].max()
    box = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
    flat = " ".join(f"{v:.6f}" for row in pts for v in row)
    return f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {flat}"


def main():
    """Converts the JSON keypoint dataset to Ultralytics pose format."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", default="court-pose-dataset")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--img-size", type=int, nargs=2, default=[1280, 720])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    samples = json.loads(Path(args.data_json).read_text())
    random.Random(args.seed).shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    splits = {"val": samples[:n_val], "train": samples[n_val:]}

    out = Path(args.out)
    w, h = args.img_size
    kept = 0
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for s in items:
            label = convert(s["kps"], w, h)
            if label is None:
                continue
            img = Path(args.images_dir) / f"{s['id']}.png"
            if not img.exists():
                continue
            shutil.copy(img, out / "images" / split / img.name)
            (out / "labels" / split / f"{s['id']}.txt").write_text(label + "\n")
            kept += 1

    (out / "court-pose.yaml").write_text(YAML.format(root=out.resolve()))
    print(f"wrote {kept} samples to {out}")


if __name__ == "__main__":
    main()
