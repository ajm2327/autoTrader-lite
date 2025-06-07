# Dockerfile
FROM python:3.12.3-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Inatall dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Expore port for streamlit
EXPOSE 8501

#DEFAULT command
CMD ["python", "historical_data_simulator.py"]