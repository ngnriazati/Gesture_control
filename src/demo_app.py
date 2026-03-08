import streamlit as st
import requests
import json

st.set_page_config(page_title="Gesture Recognition Demo", layout="centered")

st.title("🎮 Gesture Recognition Inference Demo")
st.write("This demo sends simulated landmark data to the FastAPI backend running on Kubernetes.")

# Replace with your local or minikube service URL
API_URL = st.text_input("API Endpoint:", "http://127.0.0.1:53552/predict")

# Predefined test gestures (simulated landmark patterns)
gestures = {
    "FIST": [0.10,0.20,0,0.11,0.21,0,0.12,0.22,0,0.13,0.23,0,0.14,0.24,0,
             0.15,0.25,0,0.16,0.26,0,0.17,0.27,0,0.18,0.28,0,0.19,0.29,0,
             0.20,0.30,0,0.21,0.31,0,0.22,0.32,0,0.23,0.33,0,0.24,0.34,0,
             0.25,0.35,0,0.26,0.36,0,0.27,0.37,0,0.28,0.38,0,0.29,0.39,0,
             0.30,0.40,0],
}

gesture_choice = st.selectbox("🖐️ Choose a gesture to test:", list(gestures.keys()))

if st.button("Send to API 🚀"):
    vals = gestures[gesture_choice]
    payload = {"vals": vals}

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ API Response: {result}")
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"⚠️ Could not reach API: {e}")

st.markdown("---")
st.caption("Backend: FastAPI + ML model (deployed on Kubernetes).")

