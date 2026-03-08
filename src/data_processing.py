import pandas as pd
import numpy as np
import os

RAW = "/Users/negin/Desktop/gusture_control/data/raw/landmarks_log.csv"
OUT = "/Users/negin/Desktop/gusture_control/data/processed/feat.csv"
os.makedirs("data/processed", exist_ok=True)

# Load the raw data
df = pd.read_csv(RAW)
print("Loaded:", RAW, "rows:", len(df))

# --- FEATURE EXTRACTION ---
# We'll compute simple geometric features invariant to hand size and position
def feats_from_row(r):
    xs = np.array([r[f"x{i}"] for i in range(21)], dtype=np.float32)
    ys = np.array([r[f"y{i}"] for i in range(21)], dtype=np.float32)

    # scale by hand size (width/height)
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    scl = max(np.hypot(w, h), 1e-6)

    # distances fingertip→MCP normalized
    idx = np.hypot(xs[8] - xs[5], ys[8] - ys[5]) / scl
    mid = np.hypot(xs[12] - xs[9], ys[12] - ys[9]) / scl
    rng = np.hypot(xs[16] - xs[13], ys[16] - ys[13]) / scl
    pnk = np.hypot(xs[20] - xs[17], ys[20] - ys[17]) / scl

    # index horizontal direction (tip vs base)
    idx_dx = xs[8] - xs[5]

    # finger up flags (y-tip < y-pip)
    idx_up = 1.0 if ys[8] < ys[6] - 0.01 else 0.0
    mid_up = 1.0 if ys[12] < ys[10] - 0.01 else 0.0
    rng_up = 1.0 if ys[16] < ys[14] - 0.01 else 0.0
    pnk_up = 1.0 if ys[20] < ys[18] - 0.01 else 0.0

    return [idx, mid, rng, pnk, idx_dx, idx_up, mid_up, rng_up, pnk_up]

# Process each row
rows = []
for _, r in df.iterrows():
    rows.append(feats_from_row(r) + [r["gesture"]])

feat_cols = [
    "dist_idx", "dist_mid", "dist_rng", "dist_pnk",
    "idx_dx", "idx_up", "mid_up", "rng_up", "pnk_up", "gesture"
]
feat = pd.DataFrame(rows, columns=feat_cols)

# Filter only valid gesture classes
valid_gestures = ["OPEN", "FIST", "POINT_LEFT", "POINT_RIGHT", "SHOOT"]
feat = feat[feat["gesture"].isin(valid_gestures)].reset_index(drop=True)

feat.to_csv(OUT, index=False)
print("✅ Saved:", OUT, "rows:", len(feat))
