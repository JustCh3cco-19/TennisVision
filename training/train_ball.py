"""Fine-tune YOLO26 on the local tennis-ball detection dataset.

The default dataset path points to the extracted local archive:
    datasets/ball/data.yaml

For the report: run this with --model yolo26n.pt and yolo26s.pt and compare
mAP50/mAP50-95 and inference speed against the original baseline.
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/ball/data.yaml", help="path to data.yaml")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--resume", default=None, metavar="LAST_PT",
                   help="path to a last.pt checkpoint to resume from")
    args = p.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                              batch=args.batch, project="runs/ball",
                              name=args.model.split(".")[0])

    summary = {"model": args.model, "epochs": args.epochs,
               "imgsz": args.imgsz, "batch": args.batch}
    for split in ("val", "test"):
        metrics = model.val(split=split)
        summary[split] = {
            "mAP50": round(metrics.box.map50, 4),
            "mAP50-95": round(metrics.box.map, 4),
            "precision": round(metrics.box.mp, 4),
            "recall": round(metrics.box.mr, 4),
            "inference_ms_per_img": round(metrics.speed["inference"], 2),
        }
        print(f"[{split}] mAP50: {metrics.box.map50:.4f}  "
              f"mAP50-95: {metrics.box.map:.4f}  "
              f"inference: {metrics.speed['inference']:.1f} ms/img")

    out = Path(results.save_dir) / "final_metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"metrics saved to {out}")


if __name__ == "__main__":
    main()
