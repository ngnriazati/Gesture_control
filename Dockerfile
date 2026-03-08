# ---------- Base Image ----------
FROM python:3.11-slim

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Install System Dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# ---------- Copy Requirements ----------
COPY requirements.txt .

# ---------- Install Python Dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Project Code ----------
COPY . .

# ---------- Expose Port ----------
EXPOSE 8000

# ---------- Run FastAPI App ----------
CMD ["uvicorn", "Inference_app:app", "--host", "0.0.0.0", "--port", "8000"]

