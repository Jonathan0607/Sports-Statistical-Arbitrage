# ============================================================================
# StatArb Engine — Production Container
# Multi-stage / optimized Python 3.11 image with native Eigen C++ compilation
# ============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# 1. Install system build dependencies and runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    postgresql-client \
    libeigen3-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pybind11 \
    && pip install --no-cache-dir -r requirements.txt

# 3. Copy C++ core sources and compile bare-metal quant_core extension
COPY cpp_core/ cpp_core/
COPY CMakeLists.txt .
COPY build_core.sh .
RUN chmod +x build_core.sh && ./build_core.sh

# 4. Copy project application code
COPY infrastructure/ infrastructure/
COPY src/ src/
COPY scrapers/ scrapers/
COPY models/ models/
COPY tests/ tests/

# Default command: start high-frequency poller engine
CMD ["python", "scrapers/hfp_poller.py"]
