cmake_minimum_required(VERSION 3.0.0)
project(Harris VERSION 0.1.0)

include(FetchContent)

# FetchContent_Declare(
#   GoogleBenchmark
#   URL https://github.com/google/benchmark/archive/v1.5.2.tar.gz
#   )

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  )


# if (NOT GoogleBenchmark_POPULATED)
#   FetchContent_Populate(GoogleBenchmark)
#   set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
#   set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
#   add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
# endif()

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

include_directories(${PROJECT_SOURCE_DIR})
add_executable(Harris src/main.cpp src/CPU/harris.cpp)
target_compile_features(Harris PUBLIC cxx_std_17)
target_link_libraries( Harris ${PNG_LIBRARY} CLI11::CLI11 spdlog::spdlog)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})