"""
compare_pred_gt.py
------------------
Compares model predictions (green) vs ground-truth keypoints (red)
and saves visualization images.
"""

import os, json
import numpy as np
from PIL import Image, ImageDraw
import torch
from train import TinyPoseNet

def draw_comparison(img, gt_pts, pred_pts, save_path):
    """Draws ground-truth (red) and predicted (green) dots on image."""
    w, h = img.size
    d = ImageDraw.Draw(img)

    # draw ground truth keypoints (red)
    for x, y in gt_pts:
        d.ellipse((x*w-3, y*h-3, x*w+3, y*h+3), outline=(255,0,0), width=2)

    # draw predicted keypoints (green)
    for x, y in pred_pts:
        d.ellipse((x*w-3, y*h-3, x*w+3, y*h+3), outline=(0,255,0), width=2)

    img.save(save_path)

def main():
    ann_path = "data/processed/annotations.json"
    model_path = "models/best_model.pt"
    out_dir = "artifacts/compare"
    os.makedirs(out_dir, exist_ok=True)

    # load annotations and model
    data = json.load(open(ann_path))
    model = TinyPoseNet(24)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"Loaded {len(data)} images for comparison")

    for i, rec in enumerate(data[:6]):  # compare first 6
        img = Image.open(rec["image_path"]).convert("RGB").resize((128,128))
        gt = np.array(rec["keypoints_norm"])

        # prepare input for model
        x = torch.tensor(np.array(img)).permute(2,0,1).float().unsqueeze(0)/255.0
        with torch.no_grad():
            pred = model(x).cpu().view(-1,2).numpy()

        save_path = os.path.join(out_dir, f"compare_{i}.jpg")
        draw_comparison(img.copy(), gt, pred, save_path)
        print(f"✅ Saved comparison -> {save_path}")

    print(f"All comparison images saved in: {out_dir}")

if __name__ == "__main__":
    main()
