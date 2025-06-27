# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install git before installing requirements, as one of them is a git repository
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code into the container at /app
# This includes main.py (or your specific API entry point file)
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV MODULE_NAME="main"
ENV VARIABLE_NAME="app"

# Run uvicorn when the container launches.
# Assumes your FastAPI app instance is in 'main.py' and named 'app'.
# If your main file is, for example, 'api/server.py' and app instance is 'my_api',
# you would change this to: CMD ["uvicorn", "api.server:my_api", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "uvicorn ${MODULE_NAME}:${VARIABLE_NAME} --host 0.0.0.0 --port 8000"]
