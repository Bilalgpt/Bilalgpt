# lane-feature-extractor

lane detection from dashcam/road images using a lightweight encoder-decoder network. built this mostly for a side project where i needed to extract lane geometry from recorded drive footage — couldn't find anything that was easy to drop into a pipeline without a bunch of setup, so i wrote this.

works reasonably well on highway footage. urban roads with painted lines are hit or miss depending on lighting.

## what it does

- detects up to 6 lanes per frame using instance-level segmentation masks
- color codes each detected lane
- optionally fits a centerline polyline through each mask
- batch inference mode for processing full folders of images

## setup

```bash
pip install -r requirements.txt
```

you'll need a weights file. i trained on TuSimple but the architecture is generic enough that you can swap in your own training data pretty easily.

## usage

single image:
```python
from lane_detector import LaneDetector
from utils.visualize import overlay_lanes

detector = LaneDetector(weights_path="./weights/lanenet.pth")
import cv2
img = cv2.imread("road.jpg")
masks, confidences = detector.detect(img)
result = overlay_lanes(img, masks, confidences)
cv2.imwrite("result.jpg", result)
```

batch mode over a folder:
```bash
python batch_infer.py --input ./data/images --output ./output --mode both --equalize
```

`--equalize` runs CLAHE preprocessing first which helps a lot in low-light conditions.

## config

edit `config.py` to change thresholds, colors, image size etc. main ones:

| param | default | what it does |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.45 | mask confidence cutoff per lane |
| `IMG_HEIGHT / IMG_WIDTH` | 256 / 512 | input resolution |
| `NUM_LANES_MAX` | 6 | max lanes to detect |
| `LINE_THICKNESS` | 3 | centerline draw width |

## structure

```
lane-feature-extractor/
├── lane_detector.py       # main detector class + model definition
├── batch_infer.py         # folder-level batch processing
├── config.py              # all tuneable params in one place
├── requirements.txt
└── utils/
    ├── preprocessing.py   # CLAHE, ROI crop, normalization
    └── visualize.py       # overlay and centerline drawing
```

## notes / known issues

- the decoder upsampling is bilinear which causes some blurriness at lane boundaries. tried transposed convs, actually slightly better, left it in
- very faint lane markings (worn paint) often get missed below 0.45 confidence — lowering the threshold helps but introduces more false positives near road edges
- no temporal smoothing across frames yet, so there's some flicker in video. planning to add a simple kalman filter eventually
- tested on TuSimple dataset and some of my own recorded footage from a dashcam mounted at windshield height. haven't tried drone footage or anything like that

## dataset

trained/tested on [TuSimple Lane Detection Dataset](https://github.com/TuSimple/tusimple-benchmark). you'll need to download it separately.

## todo

- [ ] batched forward pass (right now it's one image at a time despite being called "batch")
- [ ] kalman filter for temporal smoothing
- [ ] ONNX export for faster inference
- [ ] night driving fine-tuning
