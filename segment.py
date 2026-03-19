"""
segment.py — run inference on a single image or folder

usage:
    python segment.py --input road.jpg --weights ./runs/best.pth --config config.yaml
    python segment.py --input ./data/test --weights ./runs/best.pth --config config.yaml --save-dir ./output
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset import RoadwayDataset, CLASS_COLORS, get_val_transforms
import segmentation_models_pytorch as smp
from train import build_model


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--save-dir", default="./output")
    p.add_argument("--blend", type=float, default=0.5)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_model(cfg, weights_path, device):
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def infer_image(model, img_bgr, cfg, device):
    t_cfg = cfg["training"]
    h, w = t_cfg["img_height"], t_cfg["img_width"]
    transforms = get_val_transforms(h, w)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transforms(image=img_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # resize back to original resolution
    orig_h, orig_w = img_bgr.shape[:2]
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return pred


def blend_overlay(img_bgr, pred_mask, alpha=0.5):
    color_mask = RoadwayDataset.mask_to_rgb(pred_mask)
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(img_bgr, 1 - alpha, color_mask_bgr, alpha, 0)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(cfg, args.weights, device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    else:
        files = [p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED]

    for fp in tqdm(files):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        pred = infer_image(model, img, cfg, device)
        blended = blend_overlay(img, pred, alpha=args.blend)

        out_path = save_dir / fp.name
        cv2.imwrite(str(out_path), blended)

        if args.show:
            cv2.imshow("result", blended)
            cv2.waitKey(0)

    print(f"saved results to {save_dir}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
