"""Fine-tune YOLO26-pose to detect the court's 14 keypoints.

    python training/train_court_pose.py --data court-pose-dataset/court-pose.yaml

For the report: compare keypoint MAE and inference time against the original
approach (ResNet50 + direct regression).
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    """Trains the court keypoint model and saves final metrics as JSON."""
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolo26n-pose.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--resume", default=None, metavar="LAST_PT",
                   help="path to a last.pt checkpoint to resume from")
    args = p.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                              batch=args.batch, project="runs/court",
                              name=args.model.split(".")[0])
    metrics = model.val()
    print(f"pose mAP50: {metrics.pose.map50:.4f}  "
          f"pose mAP50-95: {metrics.pose.map:.4f}  "
          f"inference: {metrics.speed['inference']:.1f} ms/img")

    summary = {"model": args.model, "epochs": args.epochs,
               "imgsz": args.imgsz, "batch": args.batch,
               "val": {
                   "pose_mAP50": round(metrics.pose.map50, 4),
                   "pose_mAP50-95": round(metrics.pose.map, 4),
                   "inference_ms_per_img": round(metrics.speed["inference"], 2),
               }}
    out = Path(results.save_dir) / "final_metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"metrics saved to {out}")


if __name__ == "__main__":
    main()
