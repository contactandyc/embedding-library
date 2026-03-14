# SPDX-FileCopyrightText: 2026 Andy Curtis <contactandyc@gmail.com>
# SPDX-FileCopyrightText: 2024–2025 Knode.ai — technical questions: contact Andy (above)
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# A small "apps" project that builds example/demo binaries.
project(embedding-library_demo LANGUAGES C)

# Optional math lib on POSIX
find_library(M_LIB m)

# Bring in suite-level dependencies derived by the renderer
find_package(a_json_library CONFIG REQUIRED)
find_package(the_io_library CONFIG REQUIRED)
find_package(embedding_library CONFIG REQUIRED)
find_package(a_memory_library CONFIG REQUIRED)

# ---- Binaries ----
add_executable(parse_sentences  src/parse_sentences.c)

target_include_directories(parse_sentences PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Language standards
set_target_properties(parse_sentences PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED YES
)
if("CXX" IN_LIST CMAKE_PROJECT_LANGUAGES)
  set_target_properties(parse_sentences PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
  )
endif()

# Link the union of deps computed for this app context
target_link_libraries(parse_sentences PRIVATE  m  a_json_library::a_json_library  the_io_library::the_io_library  embedding_library::embedding_library  a_memory_library::a_memory_library)

# Optional libm
if(M_LIB)
  target_link_libraries(parse_sentences PRIVATE ${M_LIB})
endif()

# Warnings
if(MSVC)
  target_compile_options(parse_sentences PRIVATE /W4)
else()
  target_compile_options(parse_sentences PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install each app binary (handy for packaging examples)
install(TARGETS parse_sentences RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
