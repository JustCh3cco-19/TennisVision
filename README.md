# CourtVision - Tennis Match Analysis with YOLO26

Analyzes broadcast tennis videos: detects the two players and the ball,
reconstructs the court geometry, and computes shot speed, shot count,
shot type and player movement speed, rendered on an annotated video
(detection confidences, fading ball trail, top-down minimap with bounce
marks, per-player stats panel, court model overlay reprojected through the
fitted homography with RANSAC inliers/outliers highlighted).

## Pipeline

video -> court keypoints (YOLO26-pose) -> per-frame homography -> player tracking
(YOLO26) -> ball detection (YOLO26 fine-tuned) -> off-court + static-hotspot
rejection -> piecewise parabolic smoothing -> projection to metric court space
-> shot + bounce detection ->
shot classification (serve/volley/groundstroke) -> statistics -> rendering.

## Repository structure

```text
main.py                  entry point: full analysis pipeline on a video
tennisvision/            library code
    detect/              YOLO wrappers: ball, court keypoints, player tracking
    geometry.py          court reference + homography (pixel <-> metric court)
    smoothing.py         ball trajectory smoothing (piecewise parabolic fit)
    events.py            shot/hit and bounce detection
    analytics.py         shot classification, shot and movement statistics
    viz.py               frame annotations, stats panel and top-down minimap
    video.py             video I/O
training/                training and dataset-preparation scripts
    train_ball.py        fine-tune YOLO26 on the ball dataset
    train_court_pose.py  train YOLO26-pose on court keypoints
    prepare_court_pose_dataset.py  convert the court keypoint JSON dataset to Ultralytics pose format
models/                  weights (gitignored)
    court_pose_yolo26.pt       court keypoint model (best, trained on Colab)
    court_pose_yolo26_last.pt  last checkpoint (for resuming training)
    ball_yolo26n.pt / ball_yolo26s.pt  fine-tuned ball detectors
    yolo26n.pt / yolo26s.pt    pretrained base weights
    yolo26x.pt                 pretrained weights for player tracking
datasets/                datasets (gitignored)
    ball/                ball detection dataset used by train_ball.py
    ball-full/           superset of ball/ (original full export)
    court-kps/           large court keypoint dataset (used for the Colab training)
    archives/            original .zip archives
input/                   input videos (gitignored)
output/                  annotated videos and previews (gitignored)
runs/                    Ultralytics training/validation runs (gitignored)
exam/                    report and presentation material
    A - Relation/        LaTeX report
        Makefile         `make` builds diagram + report, `make clean` removes intermediates
        src/             LaTeX sources: report .tex, references.bib
            figures/     report figures (TikZ pipeline diagram + training-curve PNGs)
        pdf/             compiled report PDF
    B - Presentation/    presentation slides
```

## Install

Use a Python version supported by your local PyTorch CUDA setup, for example Python 3.12.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
python main.py --video input/match.mp4 \
    --ball-model models/ball_yolo26s.pt \
    --court-model models/court_pose_yolo26.pt \
    --output output/annotated.mp4 \
    --cache .cache
```

The terminal reports the six pipeline stages and shows frame progress,
percentage, elapsed time and ETA for long-running operations. Cache hits and
misses are printed explicitly, so it is clear whether a detector is running or
loading saved results.

The annotated output overlays the canonical court model reprojected through
the fitted per-frame homography: when the homography is correct, the projected
lines coincide with the real court lines. Detected court keypoints are colored
by RANSAC status (green inliers kept for the fit, red outliers rejected),
making the robustness of the homography estimation directly verifiable by eye.

Court keypoints are stabilized with a centered temporal median before fitting
one homography per visible frame. This follows broadcast-camera pans and zooms
without reintroducing detector jitter. `--court-smooth-window` controls the
window size (default `11` frames), while `--court-min-keypoints` rejects
close-ups and replay frames with only a few accidental keypoint detections.

Player detections are assigned independently to the near and far court halves,
so tracker ID changes do not drop a player for the rest of a rally. Short
bounded gaps are interpolated; configure the limit with `--player-max-gap`
(default `15` frames). On difficult grass/clay footage,
`--player-court-crops` reruns detection only on a court half where the
full-frame pass found no baseline player. It improves recall but costs extra
inference time. Ball inference defaults to `--ball-imgsz 1280`, which retains
more detail for the few-pixel ball on grass and clay; before smoothing,
off-court detections and static hotspots (a fixed false "ball" from on-screen
graphics or the fence, common on hardcourt) are rejected automatically so they
cannot break a real trajectory near impact. For the court itself,
`--court-preprocess` CLAHE-normalizes each frame before keypoint detection, and
`--court-refine` snaps the projected court lines onto the detected white lines
(chamfer ICP) to correct residual overlay drift. These mitigations improve
short detection failures, but long surface-specific gaps still require
fine-tuning with representative grass/clay footage.

`--ransac-thresh` sets the RANSAC reprojection threshold for the court
homography, in court meters (default `0.4`). Lower it to reject more keypoints
(stricter fit), raise it to tolerate noisier keypoint detections:

```bash
python main.py --video input/match.mp4 \
    --ball-model models/ball_yolo26s.pt \
    --court-model models/court_pose_yolo26.pt \
    --ransac-thresh 0.5
```

## Local Dataset

The ball dataset is local. The extracted training copy is:

```text
datasets/ball/data.yaml
```

A larger, unfiltered export of the same dataset is available at
`datasets/ball-full/data.yaml`. Both YAML files use only local paths.

## Training

```bash
# ball detector, default dataset: datasets/ball/data.yaml
python training/train_ball.py --model models/yolo26n.pt

# equivalent explicit command
python training/train_ball.py --data datasets/ball/data.yaml --model models/yolo26n.pt

# court keypoints: convert the local JSON dataset, then train (done on Colab)
python training/prepare_court_pose_dataset.py --data-json datasets/court-kps/data/data_train.json \
    --images-dir datasets/court-kps/data/images
python training/train_court_pose.py --data <prepared-dataset>/court-pose.yaml
```

For the GTX 1650 4 GB, start with a small batch:

```bash
python training/train_ball.py --model models/yolo26n.pt --batch 4
```
