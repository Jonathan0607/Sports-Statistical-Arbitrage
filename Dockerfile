# ============================================================================
# StatArb Engine — Multi-Stage Production Container
# Builder: compiles C++20 Eigen extension with strict flags
# Runtime: slim Python 3.11 FastAPI + uvicorn inference server
# ============================================================================

FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build_src

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install pybind11 \
    && pip install -r requirements.txt

COPY cpp_core/ cpp_core/
COPY CMakeLists.txt .

RUN PYBIND_CMAKE_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
    && cmake -B build \
        -Dpybind11_DIR="$PYBIND_CMAKE_DIR" \
        -DPYTHON_EXECUTABLE="/opt/venv/bin/python" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-std=c++20 -O3 -Wall -Wextra -Wpedantic -Werror" \
    && cmake --build build --parallel "$(nproc)" \
    && mkdir -p /opt/quant_ext \
    && cp build/quant_core*.so /opt/quant_ext/


FROM python:3.11-slim AS runtime

ARG GIT_SHA=unknown

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_DIR=models/wf_fold2 \
    GIT_SHA=${GIT_SHA}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY infrastructure/ infrastructure/
COPY config/ config/
COPY src/ src/
COPY scrapers/ scrapers/
COPY models/ models/
COPY --from=builder /opt/quant_ext/quant_core*.so /app/src/

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
