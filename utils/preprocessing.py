import cv2
import numpy as np


def clahe_equalize(img_bgr):
    """apply CLAHE to the L channel in LAB space — helps a lot in low light"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def crop_roi(img_bgr, top_fraction=0.4):
    """crop out the sky/upper portion, lanes are in the lower part"""
    h = img_bgr.shape[0]
    start = int(h * top_fraction)
    return img_bgr[start:, :]


def normalize(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img


def pad_to_multiple(img, multiple=32):
    """pad image so H and W are divisible by `multiple`"""
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return padded, (pad_h, pad_w)
