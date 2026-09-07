cmake_minimum_required(VERSION 3.12)
project(quant_core_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compiler optimizations
set(CMAKE_BUILD_TYPE Release)

# Find pybind11 package
find_package(pybind11 REQUIRED)

# Find Eigen3 (header-only linear algebra library)
find_package(Eigen3 REQUIRED)

# Add pybind11 module target
pybind11_add_module(quant_core 
    cpp_core/shins_method.cpp 
    cpp_core/kelly_optimizer.cpp
    cpp_core/nb_loss.cpp
    cpp_core/bindings.cpp
)

# Link Eigen3 headers
target_link_libraries(quant_core PRIVATE Eigen3::Eigen)

# Enforce compiler optimization flags
target_compile_options(quant_core PRIVATE
    -O3
    -march=native
    -ffast-math
)
