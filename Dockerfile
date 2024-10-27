# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies required by Prophet, Pillow, and other Python libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    wget \
    gfortran \
    liblapack-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the Python dependencies from requirements.txt
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose the port that Hugging Face Spaces uses (7860)
EXPOSE 7860

# Set environment variables for Streamlit to run on port 7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Run the Streamlit app
CMD ["streamlit", "run", "final_demo_multi_page.py"]
