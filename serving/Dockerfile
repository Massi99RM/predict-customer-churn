FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# libgomp1 is required by LightGBM for OpenMP support.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt

# Copy app code
COPY serving/app.py .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]