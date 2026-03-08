"""
data_prep.py
------------
Reads images and auto-labels 2D pose keypoints using MediaPipe.
Saves 12 selected joints' (x,y) as normalized coordinates in annotations.json.
Also draws the detected keypoints on top of each image and saves
the visualization into artifacts/annotated/.
"""

import os, json, argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

# -----------------------------------------------
# 1️⃣ Keypoint selection indices
# -----------------------------------------------
SEL_IDXS = [0, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 32]
# 0 = nose; 11/12 = shoulders; 13/14 = elbows; 23/24 = hips;
# 25/26 = knees; 27/28 = ankles; 32 = heel

# -----------------------------------------------
# 2️⃣ Pose extraction function
# -----------------------------------------------
def extract_keypoints(image_bgr: np.ndarray):
    """Run MediaPipe Pose on one image. Returns (12,2) normalized keypoints or None."""
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                      enable_segmentation=False, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        all_pts = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
        return np.array(all_pts, dtype=np.float32)[SEL_IDXS, :]

# -----------------------------------------------
# 3️⃣ Draw keypoints function
# -----------------------------------------------
def draw_keypoints(image_bgr: np.ndarray, keypoints: np.ndarray):
    """Draw small green circles for each keypoint on a copy of the image."""
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()
    for (x, y) in keypoints:
        cx, cy = int(x * w), int(y * h)
        cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), 2)
    return annotated

# -----------------------------------------------
# 4️⃣ Main script
# -----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with raw images (*.jpg, *.png)")
    ap.add_argument("--output_dir", required=True, help="Folder where annotations.json will be written")
    ap.add_argument("--max_images", type=int, default=10, help="Limit number of images processed")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = Path("artifacts/annotated")
    vis_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in Path(args.input_dir).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    images = images[:args.max_images]

    records = []
    for p in images:
        print(f"Processing: {p}")
        img = cv2.imread(str(p))
        if img is None:
            print(f"⚠️ Could not read image: {p}")
            continue

        k = extract_keypoints(img)
        if k is None:
            print(f"⚠️ No pose detected: {p}")
            continue

        h, w = img.shape[:2]
        print(f"✅ Detected pose in {p} with size {w}x{h}")

        # Save record
        records.append({
            "image_path": str(p.resolve()),
            "width": w,
            "height": h,
            "keypoints_norm": k.tolist()
        })

        # Draw keypoints and save visualization
        annotated_img = draw_keypoints(img, k)
        vis_path = vis_dir / f"annotated_{p.stem}.jpg"
        cv2.imwrite(str(vis_path), annotated_img)
        print(f"🖼️ Saved visualization -> {vis_path}")

    # Write annotations JSON
    out_path = Path(args.output_dir) / "annotations.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n✅ Wrote {len(records)} samples -> {out_path}")
    print(f"📁 Annotated images saved in: {vis_dir}")

# -----------------------------------------------
if __name__ == "__main__":
    main()
