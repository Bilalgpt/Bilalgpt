import os

# paths
DATA_DIR = os.environ.get("LANE_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("LANE_OUTPUT_DIR", "./output")
MODEL_WEIGHTS = os.environ.get("LANE_MODEL_WEIGHTS", "./weights/lanenet.pth")

# model
IMG_HEIGHT = 256
IMG_WIDTH = 512
NUM_LANES_MAX = 6

# inference
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.35
BATCH_SIZE = 8

# visualization
LANE_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]
LINE_THICKNESS = 3
SHOW_CONFIDENCE = True
