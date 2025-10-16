cmake_minimum_required(VERSION 3.18)
project(raytracingdemo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Release flags by default if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FAST_SINGLE_PASS "Single pass ray + shade" ON)
option(USE_DIR_CACHE "Cache per-pixel ray direction vectors" ON)
option(ENABLE_TIMINGS "Print per-frame timing info" ON)

add_executable(raytracingdemo
        src/main.cpp
        src/color.hpp
        src/primitives/ray.hpp
        src/camera.hpp
        src/primitives/sphere.hpp
        src/primitives/vector3.hpp
        src/bvh.hpp
        src/aabb.hpp
        lib/OBJ_Loader.h
        src/object_loader.hpp
        src/primitives/triangle.hpp
        src/primitives/primitive.hpp
)

if(FAST_SINGLE_PASS)
    target_compile_definitions(raytracingdemo PRIVATE FAST_SINGLE_PASS=1)
endif()

if(USE_DIR_CACHE)
    target_compile_definitions(raytracingdemo PRIVATE USE_DIR_CACHE=1)
endif()

if(ENABLE_TIMINGS)
    target_compile_definitions(raytracingdemo PRIVATE ENABLE_TIMINGS=1)
endif()

# --- GLFW (needs libglfw3-dev installed) ---
find_package(glfw3 3.3 REQUIRED)

# --- OpenGL (on Linux this provides OpenGL::GL) ---
find_package(OpenGL REQUIRED)

# --- GLAD v0.1.36 via FetchContent (robust: we build our own target) ---
include(FetchContent)
FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG        v0.1.36
)
FetchContent_MakeAvailable(glad)

# Try OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(raytracingdemo PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(raytracingdemo PRIVATE USE_OMP=1)
endif()

# Extra optimization flags for GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(raytracingdemo PRIVATE -O3 -march=native -ffast-math -fno-math-errno -fno-trapping-math)
endif()

# MSVC optimization flags
if(MSVC)
    target_compile_options(raytracingdemo PRIVATE /O2 /fp:fast)
endif()

# Pedantic warnings (can be toggled off if noisy)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(raytracingdemo PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# --- Link everything ---
target_link_libraries(raytracingdemo
        PRIVATE
        glfw
        OpenGL::GL
        glad
)
