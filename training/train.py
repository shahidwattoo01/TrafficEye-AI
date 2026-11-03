"""Train the 8-class TrafficEye YOLO26 model (works locally or on a Kaggle GPU notebook).

    python training/train.py --data training/data.yaml --model yolo26s.pt --epochs 120
On Kaggle:
    !pip install -q ultralytics
    !python training/train.py --data /kaggle/input/trafficeye/data.yaml --device 0,1 --project /kaggle/working
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/data.yaml")
    ap.add_argument("--model", default="yolo26s.pt", help="pretrained YOLO26 checkpoint (n/s/m/l/x)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="trafficeye_yolo26")
    ap.add_argument("--patience", type=int, default=30)
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        project=a.project, name=a.name, patience=a.patience, cos_lr=True, close_mosaic=10,
        # augmentation tuned for roadside CCTV: lighting, weather, scale and viewpoint changes
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.45, degrees=3.0, translate=0.1, scale=0.5,
        shear=1.0, perspective=0.0005, fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.1,
        plots=True, seed=42,
    )
    best = Path(model.trainer.best)
    dst = Path("weights") / "trafficeye_yolo26.pt"
    dst.parent.mkdir(exist_ok=True)
    shutil.copy(best, dst)
    print(f"Best weights copied to {dst}")


if __name__ == "__main__":
    main()
