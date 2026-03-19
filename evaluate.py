"""
evaluate.py — compute precision, recall, F1 on a labeled test set

expects a CSV with columns: image_path, label (1=crack, 0=normal)
and runs the detector on each, reporting per-threshold metrics.

usage:
    python evaluate.py --csv ./data/test_labels.csv --weights ./weights/crack_cnn.pth
"""
import argparse
import csv
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from detect import CrackDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="test CSV with image_path, label columns")
    p.add_argument("--weights", required=True)
    p.add_argument("--threshold", type=float, default=None, help="fixed threshold; if omitted, sweeps 0.1..0.9")
    p.add_argument("--enhance", action="store_true")
    p.add_argument("--out", default="eval_results.csv")
    return p.parse_args()


def load_csv(path):
    samples = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row["image_path"], int(row["label"])))
    return samples


def image_level_score(detector, img_path, enhance):
    """aggregate tile scores to image level — max score across tiles"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    dets_all = []
    from preprocess import tile_image, normalize_tile, enhance_contrast
    if enhance:
        img = enhance_contrast(img)
    from preprocess import tile_image
    tiles = tile_image(img)
    scores = [detector.score_tile(t) for t, _ in tiles]
    return max(scores) if scores else 0.0


def compute_metrics(scores, labels, threshold):
    tp = fp = fn = tn = 0
    for s, l in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == 1 and l == 1: tp += 1
        elif pred == 1 and l == 0: fp += 1
        elif pred == 0 and l == 1: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "threshold": threshold}


def main():
    args = parse_args()
    samples = load_csv(args.csv)
    detector = CrackDetector(args.weights, threshold=0.0)  # threshold=0 to get raw scores

    print(f"running on {len(samples)} images...")
    scores, labels = [], []
    for img_path, label in tqdm(samples):
        s = image_level_score(detector, img_path, args.enhance)
        if s is not None:
            scores.append(s)
            labels.append(label)

    if args.threshold is not None:
        m = compute_metrics(scores, labels, args.threshold)
        print(f"\nthreshold={m['threshold']:.2f}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}  acc={m['accuracy']:.3f}")
    else:
        thresholds = np.arange(0.1, 0.95, 0.05)
        results = [compute_metrics(scores, labels, t) for t in thresholds]
        best = max(results, key=lambda x: x["f1"])
        print(f"\nbest threshold: {best['threshold']:.2f}")
        print(f"  precision={best['precision']:.3f}  recall={best['recall']:.3f}  f1={best['f1']:.3f}  acc={best['accuracy']:.3f}")

        # save csv
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1", "accuracy"])
            writer.writeheader()
            writer.writerows(results)
        print(f"full results saved to {args.out}")


if __name__ == "__main__":
    main()
