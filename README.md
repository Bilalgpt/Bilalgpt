# sidewalk-crack-detector

CNN-based detector for cracks and surface defects on sidewalk pavement. takes street-level images, tiles them into 224x224 patches, and classifies each tile as cracked or intact. cracked tiles get flagged with bounding boxes on the original image.

i built this while doing some work on pedestrian accessibility mapping. the motivation was that a lot of sidewalk survey data is collected from imagery (google street view, city cameras, field photos) and going through it manually is slow. this is meant to help triage which sidewalks need closer inspection.

it's not perfect — very fine hairline cracks are often missed, and shadows can trigger false positives. but for obvious surface damage it works pretty well.

## how it works

1. input image → tiled into 224×224 overlapping patches
2. each patch → scored by a small CNN (0 to 1, crack probability)
3. patches above threshold → drawn as bounding boxes on original image
4. optionally: CLAHE preprocessing to improve low-contrast crack visibility

## setup

```bash
pip install -r requirements.txt
```

## usage

detect on a single image:
```bash
python detect.py --input sidewalk.jpg --weights ./weights/crack_cnn.pth
```

detect on a folder, with contrast enhancement:
```bash
python detect.py --input ./images --weights ./weights/crack_cnn.pth --output ./results --enhance
```

evaluate on a labeled test set:
```bash
python evaluate.py --csv ./data/test_labels.csv --weights ./weights/crack_cnn.pth
```

the eval script sweeps thresholds from 0.1 to 0.9 and reports precision/recall/F1 at each. useful for finding the right threshold for your data.

## data format for training

i trained on a mix of the CRACK500 dataset and some photos collected from city sidewalk inspection programs. if you want to retrain, you need tiles labeled as crack (1) or no crack (0).

for evaluation, pass a CSV with columns `image_path` and `label`.

## model

`CrackCNN` in `detect.py` — 4 conv blocks, global average pooling, 2 FC layers, sigmoid output. kept it small on purpose since it runs on every tile and needs to be fast. tried using a mobilenet backbone at one point but the latency wasn't worth the accuracy gain for this tile size.

## results (on held-out test set)

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.45 | 0.71 | 0.89 | 0.79 |
| 0.55 | 0.81 | 0.82 | 0.81 |
| 0.65 | 0.88 | 0.74 | 0.80 |

default threshold is 0.55 which gives a reasonable balance. if you're using this for triage and want fewer missed cracks, lower it to 0.45.

## known issues

- shadows from trees or poles sometimes look like cracks to the model, especially on lighter concrete surfaces
- very fine hairline cracks (< 1mm wide in the image) are often missed
- works better on darker asphalt than on light concrete (training data imbalance)
- the tile size (224px) means cracks that span tile boundaries might be partially missed — the overlap helps but doesn't fully solve it

## structure

```
sidewalk-crack-detector/
├── detect.py         # main detection script + CrackCNN model
├── preprocess.py     # image tiling and contrast enhancement
├── evaluate.py       # precision/recall evaluation on labeled data
└── requirements.txt
```

## todo

- [ ] proper training script (currently using a separate notebook for this, should clean it up)
- [ ] severity classification (hairline / moderate / severe) instead of just binary
- [ ] GIS output — export detected locations with GPS coords if input has EXIF data
- [ ] better handling of shadows as false positives
