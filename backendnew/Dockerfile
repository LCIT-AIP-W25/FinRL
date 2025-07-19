# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Expose the port your app runs on
EXPOSE 10000

# Start the Flask app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "main:app"]
