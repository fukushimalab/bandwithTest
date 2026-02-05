cmake_minimum_required(VERSION 3.18)

# Select CUDA architectures dynamically when possible.
# - If the user sets CMAKE_CUDA_ARCHITECTURES, respect it.
# - Otherwise, prefer "native" (CMake >= 3.24) to match the local GPU.
# - For older CMake, use "all-major" or a safe baseline list.
# Note: This must run before project() to override compiler defaults.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
    set(CMAKE_CUDA_ARCHITECTURES all-major)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86)
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA arch list" FORCE)
endif()

project(bandwidth_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

add_executable(global2shared global2shared.cu)
add_executable(shared2global shared2global.cu)
