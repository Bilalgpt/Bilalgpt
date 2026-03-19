# roadway-segmentation-pipeline

semantic segmentation pipeline for extracting road-level features — roads, sidewalks, lane markings, curbs, crosswalks — from street imagery. uses pretrained encoder backbones via segmentation_models_pytorch, fine-tuned on KITTI and some cityscapes samples.

started this because i needed per-pixel labels for a mapping project and the off-the-shelf models weren't giving me the class granularity i needed (most just do road vs not-road). ended up building the whole training + inference pipeline myself.

## classes

| index | class | color |
|---|---|---|
| 0 | background | black |
| 1 | road | purple |
| 2 | sidewalk | pink |
| 3 | lane marking | yellow |
| 4 | curb | light grey |
| 5 | crosswalk | green |

## setup

```bash
pip install -r requirements.txt
```

## training

```bash
python train.py --config config.yaml --data ./data/train --val ./data/val
```

edit `config.yaml` to change model architecture, encoder, learning rate etc. i've been using `unet` with `resnet34` — trains in reasonable time and gives decent results. `fpn` is slightly better on fine details like lane markings but takes longer.

checkpoints saved to `./runs/best.pth` by default.

## inference

single image:
```bash
python segment.py --input road.jpg --weights ./runs/best.pth
```

folder of images:
```bash
python segment.py --input ./data/test --weights ./runs/best.pth --save-dir ./output --blend 0.5
```

`--blend 0.5` controls how transparent the color overlay is. 0 = original image, 1 = full mask.

## data format

the dataset loader expects:
```
data/
  images/   <- RGB images (.jpg or .png)
  masks/    <- single-channel .png masks, pixel value = class index
```

trained on KITTI road dataset + a subset of Cityscapes. if you want to use your own data you'll need to remap your class indices to match the ones in `config.yaml`.

## config

`config.yaml` has everything in one place:
- model architecture + encoder choice
- class definitions
- training hyperparameters
- augmentation flags
- paths

## results

on the validation split of KITTI + cityscapes mix:
- road: ~94% pixel accuracy
- sidewalk: ~88%
- lane marking: ~79% (hardest class, thin structures)
- curb: ~82%

lane markings are the hardest because they're thin and vary a lot in condition (faded, partly covered, etc). might help to add a dedicated loss weight for that class — haven't tried yet.

## notes

- resnet34 encoder pretrained on imagenet works surprisingly well as a starting point, even for road imagery which is pretty different from imagenet
- early stopping is set to 8 epochs patience by default — found that anything less caused premature stopping on the lane marking class which is slower to converge
- if you're training on GPU with <8GB VRAM reduce batch size to 4 in the config
- the augmentation pipeline uses albumentations, which makes it easy to add more transforms if needed

## todo

- [ ] weighted class loss (lane markings need more weight)
- [ ] mIoU metric (currently just pixel accuracy which is misleading for imbalanced classes)
- [ ] ONNX export
- [ ] video inference with temporal smoothing
