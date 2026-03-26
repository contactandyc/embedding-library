# SPDX-FileCopyrightText: 2024–2026 Andy Curtis <contactandyc@gmail.com>
# SPDX-FileCopyrightText: 2024–2025 Knode.ai
# SPDX-License-Identifier: Apache-2.0
#
# Maintainer: Andy Curtis <contactandyc@gmail.com>

# CMakeLists.txt for tests
cmake_minimum_required(VERSION 3.20)

project(embedding_library_tests LANGUAGES C)

set(A_BUILD_VARIANT "debug" CACHE STRING
    "Variant to link via embedding_library::embedding_library (debug|memory|static|shared)")
set_property(CACHE A_BUILD_VARIANT PROPERTY STRINGS debug memory static shared)

option(A_ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)

find_library(M_LIB m)

# ---- Test executables ----
set(TEST_EXECUTABLES "")

enable_testing()

# ---- Coverage aggregation ----
add_custom_target(coverage_report COMMENT "Generate coverage report")

if(A_ENABLE_COVERAGE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
  include(ProcessorCount)
  ProcessorCount(NPROC)
  if(NPROC EQUAL 0)
    set(NPROC 4)
  endif()

  add_custom_command(TARGET coverage_report PRE_BUILD
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j ${NPROC}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running tests with coverage instrumentation"
  )

  find_program(LLVM_PROFDATA_EXECUTABLE llvm-profdata)
  find_program(LLVM_COV_EXECUTABLE llvm-cov)
  if(LLVM_PROFDATA_EXECUTABLE AND LLVM_COV_EXECUTABLE)
    add_custom_command(TARGET coverage_report POST_BUILD
      COMMAND ${LLVM_PROFDATA_EXECUTABLE} merge -sparse *.profraw -o default.profdata
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Merging coverage data"
    )

    # Normal compiled library path
    add_custom_command(TARGET coverage_report POST_BUILD
      COMMAND ${LLVM_COV_EXECUTABLE} show
              ../libembedding_library_${A_BUILD_VARIANT}.a
              -instr-profile=default.profdata
              -format=html -output-dir=coverage_html
              --ignore-filename-regex='/usr/.*'
              --show-regions --show-line-counts-or-regions
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generated LLVM coverage report → open coverage_html/index.html"
    )
  endif()
endif()
