"""
preprocess.py — tile large sidewalk images into fixed-size patches for inference

sidewalk images from street-level cameras tend to be high-res and the cracks
are small, so we tile them into overlapping patches before running the CNN.
overlap helps avoid cutting cracks at tile boundaries.
"""
import os
import cv2
import numpy as np
from pathlib import Path


def tile_image(img, tile_size=224, overlap=32):
    """
    split img into overlapping tiles of shape (tile_size, tile_size).
    returns list of (tile, (row_start, col_start)) tuples.
    overlap stitching is handled during result merging.
    """
    h, w = img.shape[:2]
    stride = tile_size - overlap
    tiles = []
    for r in range(0, max(1, h - overlap), stride):
        for c in range(0, max(1, w - overlap), stride):
            r_end = min(r + tile_size, h)
            c_end = min(c + tile_size, w)
            r_start = r_end - tile_size
            c_start = c_end - tile_size
            # clamp to image bounds
            r_start = max(0, r_start)
            c_start = max(0, c_start)
            tile = img[r_start:r_start + tile_size, c_start:c_start + tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = cv2.resize(tile, (tile_size, tile_size))
            tiles.append((tile, (r_start, c_start)))
    return tiles


def normalize_tile(tile_bgr):
    img = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std


def enhance_contrast(img_bgr):
    """increase crack visibility using CLAHE on the grayscale channel"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def save_tiles(img_bgr, out_dir, stem, tile_size=224, overlap=32):
    os.makedirs(out_dir, exist_ok=True)
    tiles = tile_image(img_bgr, tile_size, overlap)
    saved = []
    for i, (tile, (r, c)) in enumerate(tiles):
        name = f"{stem}_tile_{i:04d}_r{r}_c{c}.png"
        path = os.path.join(out_dir, name)
        cv2.imwrite(path, tile)
        saved.append(path)
    return saved
