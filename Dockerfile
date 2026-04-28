FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and code
COPY . .

# Run the API on port 8080 (Cloud Run's default)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
