"""
train.py — fine-tune segmentation model on roadway data

usage:
    python train.py --config config.yaml --data ./data/train --val ./data/val
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import RoadwayDataset, get_train_transforms, get_val_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data", required=True, help="path to training data dir")
    p.add_argument("--val", default=None, help="separate val dir (optional)")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    return p.parse_args()


def build_model(cfg):
    arch = cfg["model"]["architecture"]
    build_fn = {
        "unet": smp.Unet,
        "fpn": smp.FPN,
        "deeplabv3+": smp.DeepLabV3Plus,
    }.get(arch, smp.Unet)

    return build_fn(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["num_classes"],
    )


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, masks in tqdm(loader, leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == masks).sum().item()
            total += masks.numel()
    return total_loss / len(loader.dataset), correct / total


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t_cfg = cfg["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = t_cfg["img_height"], t_cfg["img_width"]

    train_ds = RoadwayDataset(args.data, get_train_transforms(h, w))

    if args.val:
        val_ds = RoadwayDataset(args.val, get_val_transforms(h, w))
    else:
        val_size = int(len(train_ds) * t_cfg["val_split"])
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=t_cfg["batch_size"], shuffle=False, num_workers=4)

    model = build_model(cfg).to(device)

    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed from {args.resume}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_cfg["epochs"])

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(1, t_cfg["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"epoch {epoch:03d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            ckpt = os.path.join(out_dir, "best.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved best checkpoint (val_acc={val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= t_cfg["early_stopping_patience"]:
                print("early stopping triggered")
                break


if __name__ == "__main__":
    main()
