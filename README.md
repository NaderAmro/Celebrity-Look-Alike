# Celebrity Look-Alike — Face Recognition System

A deep learning face recognition system trained on CelebA, deployed as a web app where users can upload a photo or use their webcam to find which of 50 known identities they most resemble.

> **Course:** Deep Learning
> **Instructor:** Dr. Tariq Bdair
> **Milestone:** 3 — Evaluation, Deployment & Presentation
> **Team:** Nader Amro, Saba Mohammad, Mira Diab

---

## What this project does

Given any face photo (uploaded or captured live from a webcam), the system:

1. **Detects the face** using MTCNN
2. **Crops it** with a 25% margin to match the training distribution
3. **Embeds it** into a 1024-dimensional space using a ConvNeXt-Small backbone
4. **Classifies** it against 50 known identities using an ArcFace classifier with Test-Time Augmentation
5. **Returns the top-K matches** along with example images of each predicted identity

The web frontend shows a side-by-side comparison: your photo on the left, the look-alike on the right.

---


Live API endpoints (when running because its in demo phase):
- **Frontend:** `http://localhost:8000/ui`
- **Swagger docs:** `http://localhost:8000/docs`

---

## Architecture

```
┌─────────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│   Browser   │───▶│  FastAPI │───▶│    MTCNN     │───▶│ ConvNeXt │
│ (HTML+JS)   │    │  Server  │    │  (detection) │    │  + Head  │
└─────────────┘    └──────────┘    └──────────────┘    └──────────┘
       ▲                                                       │
       │                                                       ▼
       │                                                ┌──────────┐
       │                                                │ ArcFace  │
       │              ┌──────────┐                      │  + TTA   │
       └──────────────│ examples │◀─────────────────────└──────────┘
                      │   /*.jpg │       top-K identities
                      └──────────┘
```

**Stack:**
- **Model:** PyTorch + timm (ConvNeXt-Small backbone)
- **Detection:** facenet-pytorch (MTCNN)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Single-file HTML + Tailwind (CDN) + vanilla JavaScript
- **Deployment:** Google Colab + ngrok for public URLs

---

## Training pipeline

The model was trained on the CelebA dataset, restricted to the 50 identities with the most images (≥30 images each).

**Key techniques:**
- **Backbone:** ConvNeXt-Small, pretrained on ImageNet
- **Loss:** ArcFace (margin=0.35, scale=64) + label smoothing
- **Embedding dim:** 1024
- **Augmentation:** RandomCrop, ColorJitter, RandomGrayscale, RandomErasing, MixUp
- **Schedule:** Warmup + cosine decay over 100 epochs (with early stopping)
- **Progressive unfreezing:** backbone frozen for first 5 epochs, then unfrozen with a 10× lower learning rate
- **Test-Time Augmentation:** 5 augmented views averaged at inference

Training script: [`pipeline.ipynb`](pipeline.ipynb)

---

## Evaluation

> ⚠️ Final metrics will be added after the evaluation pass is run. Placeholders below.

| Metric | Value |
|---|---|
| Top-1 Accuracy | 86.09% |
| Top-5 Accuracy | 97.83% |
| Macro F1 | 0.86 |
| Identities | 50 |
| Test set size | 230 images |


---

## Repository structure

```
.
├── app.py                    # FastAPI server
├── model.py                  # Inference-only model + MTCNN definitions
├── index.html                # Frontend (single-file)
├── extract_examples.py       # One-time script to build examples/
├── requirements.txt          # Pinned dependencies
├── pipeline.ipynb               # Training notebook (full pipeline)
├── checkpoints/
│   └── face_recognition_v3_final.pt   # Trained weights (~200 MB, not in git — see Setup)
├── examples/
│   └── {identity_id}.jpg     # 50 representative face images
├── results/
│   ├── cm_recognition_v3.png # Confusion matrix
│   └── history_v3.png        # Training curves
```

---

## Setup

### Option A — Run on Google Colab (recommended for grading)

1. Open `colab_bootstrap.ipynb` in Google Colab
2. Place these files in your Google Drive under `MyDrive/`:
   - `face_recognition_v3_final.pt`
   - `face_api_examples/` folder
   - `app.py`, `model.py`, `index.html`
3. Get a free ngrok auth token at [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)
4. Run the bootstrap cell (paste your ngrok token in the marked spot)
5. Open the printed `🎨 Frontend` URL

### Option B — Run locally

```bash
# Clone the repo
git clone <repo-url>
cd <repo-name>

# Set up a virtual environment
python -m venv venv
source venv/bin/activate           # macOS/Linux
# venv\Scripts\activate            # Windows

# Install dependencies
pip install -r requirements.txt

# Place the trained checkpoint (download separately) at:
# checkpoints/face_recognition_v3_final.pt

# Place the example images at:
# examples/*.jpg

# Run the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/ui** in your browser.

> 💡 **Webcam note:** browsers only allow camera access on `localhost` or HTTPS. If you serve the page from another machine over plain HTTP, the webcam tab won't work.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/info` | Model metadata (identities, embedding dim, etc.) |
| `GET` | `/ui` | Frontend HTML page |
| `GET` | `/examples/{id}.jpg` | Example image for a given identity |
| `POST` | `/predict` | Predict from a pre-cropped face image (multipart) |
| `POST` | `/predict/detect` | Predict from a full photo (runs MTCNN first) |
| `POST` | `/predict/base64` | Predict from a base64-encoded image (used by the webcam) |

All `predict*` endpoints accept `top_k` (1–100) and `use_tta` (bool) parameters.

Example response:
```json
{
  "predictions": [
    {"rank": 1, "identity": "2820", "confidence": 0.873,
     "image_url": "/examples/2820.jpg"},
    {"rank": 2, "identity": "147",  "confidence": 0.054,
     "image_url": "/examples/147.jpg"}
  ],
  "used_tta": true,
  "top_k": 5,
  "bbox": {"x1": 157, "y1": 32, "x2": 290, "y2": 202,
           "detection_confidence": 1.0}
}
```

Full interactive docs at `/docs` once the server is running.

---

## Engineering notes & challenges

This section documents real lessons from building this system — the kind of things you only learn by deploying.

**1. Pillow / facenet-pytorch dependency conflict.**
`facenet-pytorch` pins very old versions of Pillow and numpy in its `setup.py`, which silently downgrades modern Colab-preinstalled libraries and breaks torchvision. Fix: `pip install --no-deps facenet-pytorch`.

**2. CelebA is anonymized.**
The dataset's identity IDs (1–10,177) are integer labels with no celebrity names attached. We display the matched identity's example image rather than a name.

**3. Closed-set classifier limitation.**
The model recognizes exactly 50 identities. Any face — even of someone the model has never seen — will be assigned to one of those 50 with non-zero confidence. The confidence score reflects similarity within the closed set, not whether the person is "in" the dataset.

**4. Detection margin matters.**
MTCNN crops faces tightly. Our training images included hair and forehead context, so a tight crop hurt accuracy ~5%. We expand the bounding box by 25% on each side before running the recognition model.

**5. Test-Time Augmentation as free accuracy.**
Averaging the predictions of 5 augmented views (center crop, h-flip, random crop, brightness/contrast) at inference improves Top-1 by 2–3% with no retraining cost.

**6. ngrok browser warnings.**
The free tier shows an HTML warning page on first visit per browser. `<img>` tags can't click through it, which broke our example-image rendering until we added the appropriate workaround.

---

## Future work

- **Expand identity set** — retrain with hundreds or thousands of identities for a more entertaining "look-alike" experience.
- **Open-set face matching** — replace the 50-class classifier with a pretrained face-embedding model (e.g. InsightFace) and FAISS-based vector search over all of CelebA. Removes the closed-set limitation entirely.
- **Identity-to-name mapping** — manually label or scrape names for the 50 known identities to display "You look like X" instead of identity IDs.
- **Multi-image evidence** — show 3–5 example photos per matched identity in a carousel.
- **Production hardening** — Dockerize, add API key authentication, rate limiting, and proper logging.

---

## Credits

Developed for the Deep Learning course taught by **Dr. Tariq Bdair**.

**Team:**
- Nader Amro
- Saba Mohammad
- Mira Diab

Built with PyTorch, timm, FastAPI, and a lot of debugging.

---

## License

This repository is for academic submission. Not licensed for redistribution.
