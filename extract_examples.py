"""
Extract one example face image per identity from CelebA.

Run this ONCE in Colab. Saves images to ./examples/{id}.jpg.
The FastAPI app picks them up automatically at startup.

For each top_id in the checkpoint, picks the image whose face is
closest to the model's "average view" of that identity. We use the
median embedding strategy: compute embeddings for all images of that
identity, find the one whose embedding is closest to the median.
This avoids picking weird outliers (extreme angles, occlusions).

Falls back to "first image found" if anything goes wrong, so you
always get all 50 examples even if the embedding step fails.
"""
import os
import torch
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from model import load_checkpoint, infer_transform, DEVICE


def main():
    # ── Load model + checkpoint to get top_ids ─────────────
    print("[1/4] Loading model...")
    model, _, top_ids = load_checkpoint(
        "checkpoints/face_recognition_v3_final.pt"
    )
    top_set = set(top_ids)
    print(f"      {len(top_ids)} target identities")

    # ── Load CelebA and group images by identity ───────────
    print("[2/4] Loading CelebA...")
    from datasets import load_dataset
    hf = load_dataset("flwrlabs/celeba", split="train")

    # Group all images per identity (only for our 50 target IDs)
    by_id = defaultdict(list)
    for row in tqdm(hf, desc="      scanning"):
        cid = row["celeb_id"]
        if cid in top_set:
            by_id[cid].append(row["image"])

    print(f"      grouped {sum(len(v) for v in by_id.values())} "
          f"images across {len(by_id)} identities")

    # ── For each identity, pick the most "central" image ───
    print("[3/4] Selecting representative image per identity...")
    os.makedirs("examples", exist_ok=True)
    saved = 0

    for cid in tqdm(top_ids, desc="      processing"):
        imgs = by_id.get(cid, [])
        if not imgs:
            print(f"      [warn] no images for identity {cid}")
            continue

        try:
            # Compute embeddings for up to 30 images of this identity
            sample = imgs[:30]
            tensors = torch.stack([
                infer_transform(im.convert("RGB")) for im in sample
            ]).to(DEVICE)

            with torch.no_grad():
                embeddings = model(tensors)        # (N, emb_dim), already L2-normalized
                median_emb = embeddings.median(dim=0).values
                median_emb = F.normalize(median_emb.unsqueeze(0))
                # Cosine similarity to the median view
                sims = (embeddings * median_emb).sum(dim=1)
                best_idx = sims.argmax().item()

            best = sample[best_idx].convert("RGB")

        except Exception as e:
            print(f"      [warn] embedding failed for {cid}: {e}; "
                  f"falling back to first image")
            best = imgs[0].convert("RGB")

        # Resize to a reasonable display size (CelebA originals are 178x218)
        # Keep aspect ratio. 256 max edge is plenty for the frontend.
        best.thumbnail((256, 256))
        best.save(f"examples/{cid}.jpg", "JPEG", quality=88)
        saved += 1

    print(f"[4/4] Done. Saved {saved}/{len(top_ids)} example images "
          f"to ./examples/")
    print("      Restart the FastAPI server to pick them up.")


if __name__ == "__main__":
    main()
