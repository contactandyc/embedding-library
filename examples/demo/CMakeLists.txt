# SPDX-FileCopyrightText: 2024–2026 Andy Curtis <contactandyc@gmail.com>
# SPDX-FileCopyrightText: 2024–2025 Knode.ai
# SPDX-License-Identifier: Apache-2.0
#
# Maintainer: Andy Curtis <contactandyc@gmail.com>

cmake_minimum_required(VERSION 3.20)

# A dynamically generated examples project.
project(embedding_library_demo LANGUAGES C)

find_library(M_LIB m)

find_package(a_memory_library CONFIG REQUIRED)
find_package(the_io_library CONFIG REQUIRED)
find_package(a_json_library CONFIG REQUIRED)

# ---- Targets ----
add_executable(parse_sentences
  src/parse_sentences.c
)

target_include_directories(parse_sentences PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

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

# Ensure examples running in library stacks link the main library!
if(NOT TARGET embedding_library::embedding_library)
  find_package(embedding_library CONFIG REQUIRED)
endif()
target_link_libraries(parse_sentences PRIVATE embedding_library::embedding_library)

target_link_libraries(parse_sentences PRIVATE
  m
  a_memory_library::a_memory_library
  the_io_library::the_io_library
  a_json_library::a_json_library
  embedding_library::embedding_library
)

if(M_LIB)
  target_link_libraries(parse_sentences PRIVATE ${M_LIB})
endif()

if(MSVC)
  target_compile_options(parse_sentences PRIVATE /W4)
else()
  target_compile_options(parse_sentences PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS parse_sentences RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
