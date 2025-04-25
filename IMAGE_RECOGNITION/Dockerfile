# Use an official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first for faster builds
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Copy the credentials file into the container
# Make sure the credentials file is in the same directory as the Dockerfile or adjust the path accordingly
COPY credentials/image-recognition-api-457814-7ffbbd9ef309.json /credentials/image-recognition-api-457814-7ffbbd9ef309.json

# Copy .env file to the container
COPY .env .env

# Set environment variable to point to credentials (inside container)
ENV GOOGLE_APPLICATION_CREDENTIALS=/credentials/image-recognition-api-457814-7ffbbd9ef309.json

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
