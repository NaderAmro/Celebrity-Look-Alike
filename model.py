"""
Face Recognition Model + Detection — Inference Only.

Adds MTCNN-based face detection to the v3 inference pipeline.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image
from facenet_pytorch import MTCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# ArcFace (state_dict-only at inference time)
# ─────────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=64.0, m=0.35,
                 label_smoothing=0.1):
        super().__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m


# ─────────────────────────────────────────────
# Backbone + embedding head
# ─────────────────────────────────────────────
class ImprovedFaceRecognitionModel(nn.Module):
    def __init__(self, embedding_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_small", pretrained=False, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.embedding_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def get_embedding(self, x):
        features = self.backbone(x)
        emb = self.embedding_head(features)
        return F.normalize(emb, dim=1)

    def forward(self, x):
        return self.get_embedding(x)


# ─────────────────────────────────────────────
# Inference transforms
# ─────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

tta_transforms = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
]


# ─────────────────────────────────────────────
# MTCNN detector
# ─────────────────────────────────────────────
def build_detector():
    """
    MTCNN detector. We do NOT pass image_size/margin here because we want
    the raw bounding box back (so the frontend can draw it). We crop manually.
    keep_all=False → returns the single most-confident face per image.
    """
    return MTCNN(keep_all=False, device=DEVICE, post_process=False)


def detect_and_crop(detector: MTCNN, pil_image: Image.Image,
                    margin: float = 0.25) -> Tuple[Optional[Image.Image],
                                                    Optional[Tuple[int, int, int, int]],
                                                    Optional[float]]:
    """
    Detect the largest/most-confident face and return:
        (cropped_face_PIL, (x1,y1,x2,y2), confidence)

    margin expands the box by `margin` * face size on each side, since
    CelebA training images include hair/forehead context — a tight crop
    hurts accuracy.

    Returns (None, None, None) if no face is found.
    """
    img = pil_image.convert("RGB")
    boxes, probs = detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return None, None, None

    box = boxes[0]
    prob = float(probs[0]) if probs is not None else None

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * margin, h * margin

    # Expand and clamp to image bounds
    W, H = img.size
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(W, int(x2 + pad_x))
    y2 = min(H, int(y2 + pad_y))

    return img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), prob


# ─────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────
def load_checkpoint(path: str, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    model = ImprovedFaceRecognitionModel(
        embedding_dim=ckpt["embedding_dim"]
    ).to(device)
    arcface = ArcFaceLoss(
        in_features=ckpt["embedding_dim"],
        num_classes=ckpt["num_classes"],
        s=64.0, m=0.35, label_smoothing=0.1,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    arcface.load_state_dict(ckpt["arcface_state"])
    model.eval()
    arcface.eval()
    return model, arcface, ckpt["top_ids"]
