"""
FastAPI deployment for Face Recognition v3 + MTCNN + example images.

New in v3.2:
    - GET /examples/{identity}.jpg  — serves one face image per identity
    - PredictionResponse.predictions[].image_url is now populated when
      the example image exists, so the frontend can do face-to-face
      comparison ("you look like this person")

Endpoints:
    GET  /                         health check
    GET  /info                     model metadata
    GET  /ui                       frontend HTML page
    GET  /examples/{id}.jpg        example face image for an identity
    POST /predict                  pre-cropped image → prediction
    POST /predict/detect           full photo → detect face → prediction
    POST /predict/base64           base64 image → prediction (with optional detect)
"""
import io
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model import (
    load_checkpoint,
    build_detector,
    detect_and_crop,
    infer_transform,
    tta_transforms,
    DEVICE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("face-api")

MODEL_PATH = "checkpoints/face_recognition_v3_final.pt"
EXAMPLES_DIR = Path("examples")  # one image per identity: {id}.jpg
TTA_N = 5
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

state: dict = {"model": None, "arcface": None, "top_ids": None,
               "detector": None, "available_examples": set()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading model from {MODEL_PATH} (device={DEVICE})...")
    model, arcface, top_ids = load_checkpoint(MODEL_PATH)
    state["model"] = model
    state["arcface"] = arcface
    state["top_ids"] = top_ids

    logger.info("Initializing MTCNN detector...")
    state["detector"] = build_detector()

    # Index which example images exist on disk so we don't stat() per request
    if EXAMPLES_DIR.is_dir():
        available = {p.stem for p in EXAMPLES_DIR.glob("*.jpg")}
        state["available_examples"] = available
        logger.info(f"Found {len(available)} example images in "
                    f"{EXAMPLES_DIR}/")
    else:
        logger.warning(f"No {EXAMPLES_DIR}/ directory — example images "
                       "won't be served. Run extract_examples.py to "
                       "generate them.")

    logger.info(f"Ready — {arcface.num_classes} identities, "
                f"emb_dim={model.embedding_dim}")
    yield
    state.clear()


app = FastAPI(title="Face Recognition API", version="3.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ── Schemas ────────────────────────────────────────────────
class Prediction(BaseModel):
    rank: int
    identity: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    image_url: Optional[str] = None  # /examples/{id}.jpg if available


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    detection_confidence: float


class PredictionResponse(BaseModel):
    predictions: List[Prediction]
    used_tta: bool
    top_k: int
    bbox: Optional[BoundingBox] = None


class Base64Request(BaseModel):
    image_base64: str
    top_k: int = Field(5, ge=1, le=100)
    use_tta: bool = True
    detect: bool = False


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class InfoResponse(BaseModel):
    num_identities: int
    embedding_dim: int
    image_size: int
    tta_augmentations: int
    device: str
    examples_available: int


# ── Inference ──────────────────────────────────────────────
def _example_url_for(identity: str) -> Optional[str]:
    """Return /examples/{id}.jpg if we have it on disk, else None."""
    return f"/examples/{identity}.jpg" if identity in state["available_examples"] else None


@torch.no_grad()
def run_prediction(pil_image: Image.Image, top_k: int = 5,
                   use_tta: bool = True) -> List[Prediction]:
    model = state["model"]
    arcface = state["arcface"]
    top_ids = state["top_ids"]
    if model is None or arcface is None:
        raise HTTPException(503, "Model not loaded")

    img = pil_image.convert("RGB")
    weight_norm = F.normalize(arcface.weight)

    if use_tta:
        avg_probs = torch.zeros(1, arcface.num_classes, device=DEVICE)
        for tf in tta_transforms[:TTA_N]:
            tensor = tf(img).unsqueeze(0).to(DEVICE)
            embedding = model(tensor)
            cos_sim = F.linear(embedding, weight_norm) * arcface.s
            avg_probs += torch.softmax(cos_sim, dim=1)
        probs = (avg_probs / TTA_N)[0]
    else:
        tensor = infer_transform(img).unsqueeze(0).to(DEVICE)
        embedding = model(tensor)
        cos_sim = F.linear(embedding, weight_norm) * arcface.s
        probs = torch.softmax(cos_sim, dim=1)[0]

    top_k = min(top_k, arcface.num_classes)
    top_p, top_i = probs.topk(top_k)
    out = []
    for rank, (i, p) in enumerate(zip(top_i, top_p), 1):
        identity = str(top_ids[i.item()])
        out.append(Prediction(
            rank=rank,
            identity=identity,
            confidence=round(float(p.item()), 4),
            image_url=_example_url_for(identity),
        ))
    return out


def run_detect_and_predict(pil_image: Image.Image, top_k: int,
                           use_tta: bool) -> Tuple[List[Prediction], BoundingBox]:
    detector = state["detector"]
    if detector is None:
        raise HTTPException(503, "Detector not loaded")
    crop, box, det_conf = detect_and_crop(detector, pil_image)
    if crop is None:
        raise HTTPException(
            422,
            "No face detected in the image. Try a clearer, front-facing photo."
        )
    predictions = run_prediction(crop, top_k=top_k, use_tta=use_tta)
    bbox = BoundingBox(
        x1=box[0], y1=box[1], x2=box[2], y2=box[3],
        detection_confidence=round(det_conf, 4) if det_conf else 0.0,
    )
    return predictions, bbox


def _read_image_bytes(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large")
    if len(data) == 0:
        raise HTTPException(400, "Empty image payload")
    try:
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(400, "Could not decode image")


# ── Endpoints ──────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if state.get("model") is not None else "loading",
        device=str(DEVICE),
        model_loaded=state.get("model") is not None,
    )


@app.get("/info", response_model=InfoResponse)
def info():
    if state.get("arcface") is None:
        raise HTTPException(503, "Model not loaded")
    return InfoResponse(
        num_identities=state["arcface"].num_classes,
        embedding_dim=state["model"].embedding_dim,
        image_size=224,
        tta_augmentations=TTA_N,
        device=str(DEVICE),
        examples_available=len(state.get("available_examples", set())),
    )


@app.get("/ui")
def ui():
    """Serve the frontend HTML page."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "index.html not found next to app.py")
    return FileResponse(html_path)


@app.get("/examples/{filename}")
def example_image(filename: str):
    """
    Serve one face image per CelebA identity.
    Filenames look like '2820.jpg'. Defends against path traversal by
    requiring a clean basename.
    """
    safe = Path(filename).name  # strip any directory components
    if safe != filename or not safe.endswith(".jpg"):
        raise HTTPException(400, "Invalid filename")
    path = EXAMPLES_DIR / safe
    if not path.is_file():
        raise HTTPException(404, "Example not found for that identity")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=100),
    use_tta: bool = Query(True),
):
    """Predict from a pre-cropped face image (no detection)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    contents = await file.read()
    img = _read_image_bytes(contents)
    predictions = run_prediction(img, top_k=top_k, use_tta=use_tta)
    return PredictionResponse(
        predictions=predictions, used_tta=use_tta, top_k=top_k, bbox=None
    )


@app.post("/predict/detect", response_model=PredictionResponse)
async def predict_detect(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=100),
    use_tta: bool = Query(True),
):
    """Detect the face in a full photo, crop it, then predict."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    contents = await file.read()
    img = _read_image_bytes(contents)
    predictions, bbox = run_detect_and_predict(img, top_k, use_tta)
    return PredictionResponse(
        predictions=predictions, used_tta=use_tta, top_k=top_k, bbox=bbox
    )


@app.post("/predict/base64", response_model=PredictionResponse)
def predict_base64(req: Base64Request):
    """
    Predict from a base64 image. Set `detect=true` for full photos
    (e.g. webcam snapshots), `detect=false` for pre-cropped faces.
    """
    b64 = req.image_base64
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        data = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64: {e}")
    img = _read_image_bytes(data)

    if req.detect:
        predictions, bbox = run_detect_and_predict(img, req.top_k, req.use_tta)
        return PredictionResponse(
            predictions=predictions, used_tta=req.use_tta,
            top_k=req.top_k, bbox=bbox,
        )

    predictions = run_prediction(img, top_k=req.top_k, use_tta=req.use_tta)
    return PredictionResponse(
        predictions=predictions, used_tta=req.use_tta,
        top_k=req.top_k, bbox=None,
    )
