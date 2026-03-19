import cv2
import numpy as np
from config import LANE_COLORS, LINE_THICKNESS, SHOW_CONFIDENCE


def overlay_lanes(img_bgr, masks, confidences):
    """draw detected lane masks on the image with color per lane"""
    overlay = img_bgr.copy()
    for i, (mask, conf) in enumerate(zip(masks, confidences)):
        if mask.sum() == 0:
            continue
        color = LANE_COLORS[i % len(LANE_COLORS)]
        colored = np.zeros_like(img_bgr)
        colored[mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)

        if SHOW_CONFIDENCE:
            ys, xs = np.where(mask == 1)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                label = f"L{i+1}: {conf:.2f}"
                cv2.putText(
                    overlay, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )
    return overlay


def draw_centerlines(img_bgr, masks):
    """fit a polyline through each lane mask and draw it"""
    result = img_bgr.copy()
    for i, mask in enumerate(masks):
        if mask.sum() < 10:
            continue
        color = LANE_COLORS[i % len(LANE_COLORS)]
        ys, xs = np.where(mask == 1)
        if len(xs) < 5:
            continue
        # bin by y and take mean x per bin — simple but works fine
        bins = np.linspace(ys.min(), ys.max(), 20).astype(int)
        pts = []
        for j in range(len(bins) - 1):
            idx = np.where((ys >= bins[j]) & (ys < bins[j + 1]))[0]
            if len(idx) > 0:
                pts.append((int(xs[idx].mean()), int(ys[idx].mean())))
        if len(pts) >= 2:
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(result, a, b, color, LINE_THICKNESS, cv2.LINE_AA)
    return result


def save_result(img, path):
    cv2.imwrite(path, img)
    print(f"saved → {path}")
