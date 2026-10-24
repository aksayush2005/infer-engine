# LLM Inference

A C++20 GGUF inference application with AVX2/FMA kernels and an interactive chat loop.

## Structure

- `include/llm/` contains shared types, component interfaces, and inline class definitions.
- `src/` contains GGUF, tokenizer, tensor operations, attention, inference, and application sources.

## Prerequisites

- CMake 3.16 or newer
- A C++20 compiler with AVX2 and FMA support
- A compatible GGUF model file

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Run

```sh
./build/llm-inference path/to/model.gguf
```

Enter `quit` to exit the interactive prompt.
