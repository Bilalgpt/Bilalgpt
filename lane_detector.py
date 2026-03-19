import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    IMG_HEIGHT, IMG_WIDTH, CONFIDENCE_THRESHOLD,
    NUM_LANES_MAX, MODEL_WEIGHTS
)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LaneNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(ConvBNReLU(3, 64), ConvBNReLU(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.layer3 = nn.Sequential(
            ConvBNReLU(128, 256), ConvBNReLU(256, 256), ConvBNReLU(256, 256)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.layer4 = nn.Sequential(
            ConvBNReLU(256, 512), ConvBNReLU(512, 512), ConvBNReLU(512, 512)
        )

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.layer4(x)
        return x


class LaneNetDecoder(nn.Module):
    def __init__(self, num_lanes):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = ConvBNReLU(256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = ConvBNReLU(128, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = ConvBNReLU(64, 64)
        self.final = nn.Conv2d(64, num_lanes + 1, 1)

    def forward(self, x):
        x = self.conv1(self.up1(x))
        x = self.conv2(self.up2(x))
        x = self.conv3(self.up3(x))
        return self.final(x)


class LaneNet(nn.Module):
    def __init__(self, num_lanes=NUM_LANES_MAX):
        super().__init__()
        self.encoder = LaneNetEncoder()
        self.decoder = LaneNetDecoder(num_lanes)
        self.num_lanes = num_lanes

    def forward(self, x):
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits


class LaneDetector:
    def __init__(self, weights_path=MODEL_WEIGHTS, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LaneNet()
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, path):
        import os
        if not os.path.exists(path):
            print(f"[warn] weights not found at {path}, running with random init")
            return
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        print(f"loaded weights from {path}")

    def preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    def detect(self, img_bgr):
        tensor = self.preprocess(img_bgr).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        lane_probs = probs[0, 1:].cpu().numpy()  # skip background channel
        masks = (lane_probs > CONFIDENCE_THRESHOLD).astype(np.uint8)
        # resize masks back to original image size
        h, w = img_bgr.shape[:2]
        resized = []
        for m in masks:
            resized.append(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
        return resized, lane_probs.max(axis=(1, 2))
