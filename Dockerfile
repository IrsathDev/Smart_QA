# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (for PDF support)
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data folder
RUN mkdir -p data

# Expose port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]