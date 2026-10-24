cmake_minimum_required(VERSION 3.16)
project(llm-inference LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(llm-inference
  src/main.cpp
  src/gguf.cpp
  src/tokenizer.cpp
  src/ops.cpp
  src/attention.cpp
  src/inference.cpp
)

target_include_directories(llm-inference PRIVATE include)
target_compile_features(llm-inference PRIVATE cxx_std_20)
target_link_libraries(llm-inference PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(llm-inference PRIVATE -mavx2 -mfma)
endif()
