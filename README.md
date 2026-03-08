# Gesture Control – End-to-End ML & MLOps Demo

This project demonstrates a complete **machine learning and MLOps pipeline** for gesture recognition and control.  
The system detects hand gestures from a webcam, extracts features, trains a machine learning model, and deploys the model as a production-ready API using **FastAPI, Docker, and Kubernetes**.

The goal of this project is to show how a machine learning system moves from **data collection → training → deployment → scalable infrastructure**.

---

# System Architecture

Camera → MediaPipe → Landmark Features → ML Model → FastAPI API → Docker → Kubernetes

The pipeline contains:

1. Gesture dataset collection
2. Feature extraction
3. Model training
4. Experiment tracking with MLflow
5. Model serving via FastAPI
6. Containerization using Docker
7. Kubernetes deployment

---

# Project Structure


gesture_control/

Dockerfile
docker-compose.yml
requirements.txt
README.md

src/
hand_game.py
inference_app.py
trained_model.py

data/
raw/
processed/
training/

k8s/
api.yaml
mlflow.yaml
localstack.yaml


---

# Key Features

• Real-time hand gesture detection  
• Dataset collection using webcam  
• Feature extraction from hand landmarks  
• RandomForest machine learning model  
• MLflow experiment tracking  
• REST API for model inference  
• Docker containerization  
• Kubernetes deployment  

---

# Gesture Dataset Collection

The dataset is created using a **webcam-based game environment**.

The system captures hand landmarks using MediaPipe and logs them into CSV files while playing.

Example gestures detected:

• FIST  
• OPEN  
• POINT_LEFT  
• POINT_RIGHT  
• SHOOT  

Each frame records:


timestamp
hand landmark coordinates
gesture label


---

# Feature Engineering

Raw landmarks are converted into meaningful features.

Example features extracted:

• finger distance ratios  
• finger extension indicators  
• directional offsets  

These features are used as inputs for the machine learning model.

---

# Model Training

The gesture classifier is trained using **RandomForest**.

Example configuration:


RandomForestClassifier(
n_estimators=400,
max_depth=18,
class_weight="balanced"
)


Training logs the following information to MLflow:

• model parameters  
• evaluation metrics  
• trained model artifact  

The trained model is saved as:


data/training/gesture_rf.pkl


---

# API Inference Service

The trained model is deployed using **FastAPI**.

### Health Check


GET /health


Response:


{ "ok": true }


### Gesture Prediction


POST /predict


Example input:


{
"vals": [x0,y0,x1,y1,...]
}


Example response:


{
"gesture": "FIST"
}


---

# Running the Project Locally

Install dependencies:


pip install -r requirements.txt


Run the API:


uvicorn src.inference_app:app --reload


Open API documentation:


http://localhost:8000/docs


---

# Docker Deployment

Build the Docker image:


docker build -t gesture-api .


Run the container:


docker run -p 8000:8000 gesture-api


Access the API:


http://localhost:8000/docs


---

# Docker Compose Environment

This project includes a multi-service setup with:

• MLflow server  
• Gesture inference API  

Start all services:


docker-compose up


Services available:

MLflow UI


http://localhost:5050


Gesture API


http://localhost:8000/docs


---

# Kubernetes Deployment

The project includes Kubernetes manifests for deploying the full system.

Deploy everything:


kubectl apply -f k8s/


Components deployed:

• Gesture API service  
• MLflow tracking server  
• LocalStack services  

Example API service:


gesture-api-service


---

# MLflow Experiment Tracking

MLflow tracks:

• training parameters  
• training metrics  
• trained models  
• artifacts  

Open MLflow UI:


http://localhost:5050


---

# Dependencies

Main libraries used in the project:


fastapi
uvicorn
scikit-learn
pandas
numpy
mlflow
joblib
opencv-python


---

# Possible Future Improvements

• Deep learning gesture recognition model  
• ROS2 integration for robotics control  
• Cloud deployment (AWS / GCP)  
• Streaming inference pipeline  
• Real robot integration  

---

# Author

Negin

Machine Learning & Robotics Engineer

This project demonstrates **end-to-end machine learning system development and MLOps deployment** including training, experiment tracking, containerization, and orchestration.