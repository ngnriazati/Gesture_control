from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib


MODEL_PATH = 'data/training/gesture_rf.pkl'


clf = joblib.load(MODEL_PATH)

app = FastAPI(title="Gesture Inference API", version="1.0")##### because of this

# --- Define input format ---
class Landmarks(BaseModel):
    # 42 floats: x0, y0, x1, y1, ... x20, y20
    vals: list[float]

# --- Feature extraction ---
def feats_from_lm(vals):
    lm = np.array(vals, dtype=np.float32).reshape(21, 3)
    xs, ys = lm[:, 0], lm[:, 1]
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    scl = max(np.hypot(w, h), 1e-6)
    idx = np.hypot(xs[8] - xs[5], ys[8] - ys[5]) / scl
    mid = np.hypot(xs[12] - xs[9], ys[12] - ys[9]) / scl
    rng = np.hypot(xs[16] - xs[13], ys[16] - ys[13]) / scl
    pnk = np.hypot(xs[20] - xs[17], ys[20] - ys[17]) / scl
    idx_dx = xs[8] - xs[5]
    idx_up = 1.0 if ys[8] < ys[6] - 0.01 else 0.0
    mid_up = 1.0 if ys[12] < ys[10] - 0.01 else 0.0
    rng_up = 1.0 if ys[16] < ys[14] - 0.01 else 0.0
    pnk_up = 1.0 if ys[20] < ys[18] - 0.01 else 0.0
    return np.array([[idx, mid, rng, pnk, idx_dx, idx_up, mid_up, rng_up, pnk_up]], dtype=np.float32)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(lm: Landmarks):
    X = feats_from_lm(lm.vals)
    pred = clf.predict(X)[0]
    return {"gesture": pred}

