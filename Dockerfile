# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Cloud Run requires PORT env variable
ENV PORT=8080

# Run with gunicorn (production server)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app