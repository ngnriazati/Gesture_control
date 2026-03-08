import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

DATA = "/Users/negin/Desktop/igusture_control/data/processed/feat.csv"
MODEL = "/Users/negin/Desktop/gusture_control/data/training/gesture_rf.pkl"
os.makedirs("data/models", exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(DATA)
X = df[["dist_idx","dist_mid","dist_rng","dist_pnk",
        "idx_dx","idx_up","mid_up","rng_up","pnk_up"]].values
y = df["gesture"].values

# --- Split train/test ---
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- MLflow setup ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("gesture-demo")

with mlflow.start_run():
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)

    acc = accuracy_score(yte, yhat)
    print("Accuracy:", round(acc, 3))
    print(classification_report(yte, yhat))
    print("Confusion:\n", confusion_matrix(yte, yhat))

    # Log everything to MLflow
    mlflow.log_param("model", "RandomForest")
    mlflow.log_params({"n_estimators":400, "max_depth":18})
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

    # Save locally
    joblib.dump(clf, MODEL)
    mlflow.log_artifact(MODEL)
    print("✅ Saved model to:", MODEL)
