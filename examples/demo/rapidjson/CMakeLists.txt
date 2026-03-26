# SPDX-FileCopyrightText: 2024–2026 Andy Curtis <contactandyc@gmail.com>
# SPDX-FileCopyrightText: 2024–2025 Knode.ai
# SPDX-License-Identifier: Apache-2.0
#
# Maintainer: Andy Curtis <contactandyc@gmail.com>

cmake_minimum_required(VERSION 3.14)
project(rapidjson_bench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Fetch RapidJSON (Download ONLY)
include(FetchContent)
FetchContent_Declare(
    rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        master
)

# Download the code, but DO NOT run their CMakeLists.txt
FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
    FetchContent_Populate(rapidjson)
endif()

# 2. Add Executable
add_executable(rapid_bench main_rapidjson.cpp)

# 3. Link headers manually
# Since we didn't "build" the library target, we just point to the source directory
target_include_directories(rapid_bench PRIVATE ${rapidjson_SOURCE_DIR}/include)

# 4. Release Optimizations
if(MSVC)
    target_compile_options(rapid_bench PRIVATE /O2 /Ob2)
else()
    target_compile_options(rapid_bench PRIVATE -O3 -march=native)
endif()
