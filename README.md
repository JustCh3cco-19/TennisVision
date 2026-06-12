# CourtVision - Tennis Match Analysis with YOLO26

Analyzes broadcast tennis videos: detects the two players and the ball,
reconstructs the court geometry, and computes shot speed, shot count,
shot type and player movement speed, rendered on an annotated video
(detection confidences, fading ball trail, top-down minimap with bounce
marks, per-player stats panel).

## Pipeline

video -> court keypoints (YOLO26-pose) -> homography -> player tracking
(YOLO26) -> ball detection (YOLO26 fine-tuned) -> piecewise parabolic
smoothing -> projection to metric court space -> shot + bounce detection ->
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
