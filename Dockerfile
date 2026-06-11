# Use a standard Python base image
FROM python:3.11-slim

# Create a non-root user with a UID in the 10000-20000 range required by Choreo
RUN groupadd -g 10001 choreo && \
    useradd -u 10001 -g choreo -m choreouser

# Set the working directory
WORKDIR /app

# Copy dependency file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R choreouser:choreo /app

# Switch to the non-root user
USER choreouser

# Set the entry point for your application (e.g., uvicorn or python)
CMD ["python", "app.py"]
