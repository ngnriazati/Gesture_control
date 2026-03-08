"""
train.py
--------
Trains a tiny CNN to predict 12 (x,y) keypoints (24 outputs total)
from the auto-labeled data produced by data_prep.py.
Logs parameters & metrics to MLflow.
"""

import os, json, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import mlflow, mlflow.pytorch


# ---------------- Dataset ---------------- #
class PoseDataset(Dataset):
    """Loads images and their normalized keypoints from annotations.json."""
    def __init__(self, ann_path, img_size=128):
        data = json.load(open(ann_path))
        if len(data) < 2:
            raise RuntimeError("Need at least 2 labeled images.")
        self.items = data
        self.img_size = img_size

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(rec["image_path"]).convert("RGB").resize((self.img_size, self.img_size))
        x = torch.tensor(np.array(img)).permute(2,0,1).float()/255.0  # [3,H,W]
        y = torch.tensor(rec["keypoints_norm"], dtype=torch.float32).view(-1)  # [24]
        return x, y


# ---------------- Model ---------------- #
class TinyPoseNet(nn.Module):
    """A small CNN regressor."""
    def __init__(self, out_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 128→64
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64→32
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(64*4*4,256), nn.ReLU(),
            nn.Linear(256,out_dim)
        )
    def forward(self, x): return self.net(x)


# ---------------- Training ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/annotations.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    mlflow.set_experiment("vicon_mlopspose")

    ds = PoseDataset(args.data, img_size=args.img_size)
    val_n = max(1, int(len(ds)*args.val_split))
    tr_n = len(ds) - val_n
    tr_ds, va_ds = random_split(ds, [tr_n, val_n], generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyPoseNet(24).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    os.makedirs("models", exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            "lr": args.lr, "epochs": args.epochs,
            "batch_size": args.batch_size, "img_size": args.img_size
        })
        best_val = float("inf")
        for ep in range(args.epochs):
            # --- Train ---
            model.train(); train_loss = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = crit(pred, yb)
                loss.backward(); opt.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(tr_dl.dataset)

            # --- Validate ---
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = crit(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(va_dl.dataset)

            mlflow.log_metrics({"train_mse": train_loss, "val_mse": val_loss}, step=ep)
            print(f"epoch {ep+1}/{args.epochs}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_path = "models/best_model.pt"
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(best_path)

        mlflow.pytorch.log_model(model, "model")
        print("✅ Training done. Best val_mse:", best_val)

if __name__ == "__main__":
    main()
