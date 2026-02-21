FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
# Install base requirements first (tensorflow works in Linux x86_64 and aarch64).
# Then add Streamlit deps only — skip streamlit's tensorflow-cpu line (no Linux aarch64 wheel).
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "streamlit>=1.30.0" "pillow>=10.0.0" "opencv-python-headless>=4.8.0"

# Copy the rest of the project
COPY . .

# Streamlit runs on port 8501 by default
EXPOSE 8501

CMD ["streamlit", "run", "apple-variety-streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

