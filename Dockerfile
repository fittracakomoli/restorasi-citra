# Gunakan base image Python slim (ringan)
FROM python:3.12-slim

# Install dependensi sistem untuk OpenCV dan yang diperlukan
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory di container
WORKDIR /app

# Copy requirements dan install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh project ke container
COPY . .

# Build Tailwind CSS (pastikan sudah ada package.json & tailwind.config.js)
RUN apt-get update && apt-get install -y nodejs npm && \
    npm install && \
    npx tailwindcss -i ./static/css/input.css -o ./static/dist/output.css && \
    apt-get purge -y nodejs npm && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /root/.npm /root/.cache

# Ekspos port yang digunakan (Render default 10000, Railway 8080 atau sesuai)
EXPOSE 8080

# Perintah untuk menjalankan Flask via Gunicorn di container
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8080", "--workers", "2"]
