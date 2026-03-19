"""
batch_infer.py  —  run lane detection over a folder of images

usage:
    python batch_infer.py --input ./data/images --output ./output --weights ./weights/lanenet.pth

i added this because running the detector image-by-image in a loop from the
outside was painfully slow. batching the preprocessing at least helps with
throughput even though the model itself runs one image at a time here.
TODO: actual batched forward pass when time allows
"""

import argparse
import os
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from lane_detector import LaneDetector
from utils.visualize import overlay_lanes, draw_centerlines, save_result
from utils.preprocessing import clahe_equalize


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="batch lane detection over a folder")
    p.add_argument("--input", required=True, help="folder with input images")
    p.add_argument("--output", required=True, help="where to save results")
    p.add_argument("--weights", default="./weights/lanenet.pth")
    p.add_argument("--mode", choices=["overlay", "centerline", "both"], default="overlay")
    p.add_argument("--equalize", action="store_true", help="apply CLAHE before inference")
    p.add_argument("--conf", type=float, default=None, help="override confidence threshold")
    return p.parse_args()


def run(args):
    import config
    if args.conf is not None:
        config.CONFIDENCE_THRESHOLD = args.conf

    detector = LaneDetector(weights_path=args.weights)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in Path(args.input).iterdir() if p.suffix.lower() in SUPPORTED]
    if not imgs:
        print("no images found in", args.input)
        return

    print(f"processing {len(imgs)} images...")
    t0 = time.time()

    for img_path in tqdm(imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [skip] could not read {img_path.name}")
            continue

        if args.equalize:
            frame = clahe_equalize(frame)

        masks, confs = detector.detect(frame)

        if args.mode in ("overlay", "both"):
            vis = overlay_lanes(frame, masks, confs)
            save_result(vis, str(out_dir / f"overlay_{img_path.name}"))

        if args.mode in ("centerline", "both"):
            vis = draw_centerlines(frame, masks)
            save_result(vis, str(out_dir / f"centerline_{img_path.name}"))

    elapsed = time.time() - t0
    print(f"done. {len(imgs)} images in {elapsed:.1f}s ({elapsed/len(imgs):.2f}s/img)")


if __name__ == "__main__":
    run(parse_args())
