"""
detect.py — sidewalk crack/defect detection

a small CNN classifier that takes 224x224 image tiles and outputs
a defect probability. if the probability is above the threshold, the
tile is flagged and the bounding box is drawn on the original image.

usage:
    python detect.py --input sidewalk.jpg --weights ./weights/crack_cnn.pth
    python detect.py --input ./images/ --weights ./weights/crack_cnn.pth --output ./results
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from preprocess import tile_image, normalize_tile, enhance_contrast


TILE_SIZE = 224
OVERLAP = 32
DEFAULT_THRESHOLD = 0.55
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp"}


class CrackCNN(nn.Module):
    """
    nothing fancy — just a small conv net for binary classification per tile.
    tried mobilenet as backbone but this custom one is actually faster for
    this specific use case and size of tiles.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CrackDetector:
    def __init__(self, weights_path, threshold=DEFAULT_THRESHOLD, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model = CrackCNN().to(self.device)
        self._load(weights_path)
        self.model.eval()

    def _load(self, path):
        if not os.path.exists(path):
            print(f"[warn] no weights at {path}, using random init")
            return
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score_tile(self, tile_bgr):
        """returns defect probability for a single tile"""
        norm = normalize_tile(tile_bgr)
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor)
        return torch.sigmoid(logit).item()

    def detect(self, img_bgr, enhance=False):
        """
        returns list of dicts: {score, row, col, tile_size}
        only tiles above threshold are included
        """
        if enhance:
            img_bgr = enhance_contrast(img_bgr)
        tiles = tile_image(img_bgr, TILE_SIZE, OVERLAP)
        detections = []
        for tile, (r, c) in tiles:
            score = self.score_tile(tile)
            if score >= self.threshold:
                detections.append({"score": score, "row": r, "col": c, "size": TILE_SIZE})
        return detections


def draw_detections(img_bgr, detections):
    result = img_bgr.copy()
    for d in detections:
        r, c, sz = d["row"], d["col"], d["size"]
        color = (0, 0, 255) if d["score"] > 0.75 else (0, 165, 255)
        cv2.rectangle(result, (c, r), (c + sz, r + sz), color, 2)
        label = f"{d['score']:.2f}"
        cv2.putText(result, label, (c + 4, r + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output", default="./results")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--enhance", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    detector = CrackDetector(args.weights, threshold=args.threshold)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)
    files = [inp] if inp.is_file() else [p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED]

    for fp in tqdm(files):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        dets = detector.detect(img, enhance=args.enhance)
        annotated = draw_detections(img, dets)
        cv2.imwrite(str(out_dir / fp.name), annotated)
        print(f"{fp.name}: {len(dets)} defect(s) detected")


if __name__ == "__main__":
    main()
