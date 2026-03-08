"""
infer.py
--------
Uses the trained model (best_model.pt) to predict 12 keypoints (24 values)
for each image in the dataset and visualize them.
"""

import os, json, argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
from train import TinyPoseNet  # reuse your model class


def draw_kpts(img: Image.Image, kpts_norm: np.ndarray, color=(0, 255, 0)):
    """Draw keypoints on an image."""
    w, h = img.size
    d = ImageDraw.Draw(img)
    for x, y in kpts_norm:
        cx, cy = x * w, y * h
        d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=color, width=2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/processed/annotations.json",
                    help="Annotations file for test images")
    ap.add_argument("--model", default="models/best_model.pt",
                    help="Path to trained model weights")
    ap.add_argument("--out", default="artifacts/infer",
                    help="Output folder for predicted images")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- load data ---
    data = json.load(open(args.ann))
    print(f"Loaded {len(data)} images for inference")

    # --- load model ---
    model = TinyPoseNet(24)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    # --- predict and draw ---
    for i, rec in enumerate(data):
        img = Image.open(rec["image_path"]).convert("RGB").resize((128, 128))
        x = torch.tensor(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            pred = model(x).squeeze().view(-1, 2).numpy()

        # Draw predicted keypoints (green)
        out_img = draw_kpts(img.copy(), pred, color=(0, 255, 0))
        out_img.save(os.path.join(args.out, f"pred_{i}.jpg"))
        print(f"✅ Saved prediction visualization -> {args.out}/pred_{i}.jpg")

    print(f"\nAll predictions saved in {args.out}")

if __name__ == "__main__":
    main()
