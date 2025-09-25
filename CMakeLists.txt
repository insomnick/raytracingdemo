cmake_minimum_required(VERSION 3.18)
project(raytracingdemo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(raytracingdemo
        main.cpp
        color.hpp
        ray.hpp
        camera.hpp
        sphere.hpp
        vector3.hpp
)

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

# --- Link everything ---
target_link_libraries(raytracingdemo
        PRIVATE
        glfw
        OpenGL::GL
        glad
)

# Optional: warnings & handy debug flags
target_compile_options(raytracingdemo PRIVATE -Wall -Wextra -Wpedantic)
