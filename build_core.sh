#!/bin/bash
set -e

# Resolve absolute path of directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=== Compiling native C++ quant_core extension module ==="

# Resolve Python and pip executables (venv if present, otherwise system)
if [ -f "$DIR/venv/bin/python" ]; then
    PYTHON_EXE="$DIR/venv/bin/python"
    PIP_EXE="$DIR/venv/bin/pip"
else
    PYTHON_EXE="$(which python3 || which python)"
    PIP_EXE="$(which pip3 || which pip || echo "$PYTHON_EXE -m pip")"
fi

# 1. Install pybind11 if not already present
echo "[1/4] Ensuring pybind11 is installed..."
"$PIP_EXE" install pybind11

# 2. Setup build directory
echo "[2/4] Setting up build directories..."
rm -rf build
mkdir -p build
cd build

# 3. Configure with CMake using detected python path
echo "[3/4] Running CMake configuration..."
PYBIND_CMAKE_DIR=$("$PYTHON_EXE" -c "import pybind11; print(pybind11.get_cmake_dir())")

# Detect Eigen3 prefix (Homebrew on macOS or standard Linux paths)
HOMEBREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
CMAKE_PREFIXES="$HOMEBREW_PREFIX;/usr;/usr/local;/usr/include/eigen3"

cmake \
    -Dpybind11_DIR="$PYBIND_CMAKE_DIR" \
    -DPYTHON_EXECUTABLE="$PYTHON_EXE" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIXES" \
    ..

# 4. Compile C++ module
echo "[4/4] Building shared extension..."
# Use parallel cores for faster builds
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2)
make -j"$CORES"

echo "=== Copying compiled modules ==="
# Copy compiled quant_core shared objects to src/
cp quant_core*.so ../src/ 2>/dev/null || true
cp quant_core*.dylib ../src/ 2>/dev/null || true

echo "=== Native C++ core build successful ==="
