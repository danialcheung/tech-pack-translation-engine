# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set environment variables to prevent Python from buffering stdout 
# and to silence PaddleOCR/Protobuf warnings
ENV PYTHONUNBUFFERED=1
ENV GLOG_minloglevel=2
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 3. Install system dependencies required by OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Install Python dependencies first 
# (Doing this before copying the code takes advantage of Docker layer caching, 
# making future builds much faster if you only change your python script)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code (main.py, terms.json, fonts/)
COPY . .

# 7. Create the output directory inside the container
RUN mkdir -p /app/output

# 8. The execution command to run your pipeline
CMD ["python", "main.py"]