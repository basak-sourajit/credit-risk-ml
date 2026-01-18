FROM python:3.10-slim

# ----------------------------
# 1️⃣ Install system dependencies
# ----------------------------
RUN apt-get update && \
    apt-get install -y libgomp1 gcc && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------
# 2️⃣ Set working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# 3️⃣ Copy requirements first (better caching)
# ----------------------------
COPY requirements.txt .

# ----------------------------
# 4️⃣ Install Python dependencies
# ----------------------------
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 5️⃣ Copy the rest of the app
# ----------------------------
COPY . .

# ----------------------------
# 6️⃣ Set environment variables (optional but recommended)
# ----------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ----------------------------
# 7️⃣ Default command to run the API
# ----------------------------
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
