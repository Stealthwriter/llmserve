# Use an official Python runtime as a parent image
FROM python:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies directly
RUN pip install --no-cache-dir fastapi uvicorn transformers torch huggingface-hub accelerate ninja && \
    pip install --no-cache-dir --no-build-isolation flash-attn 

# Run main.py when the container launches
CMD ["python", "main.py"]
