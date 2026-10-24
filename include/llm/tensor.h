#pragma once

#include "llm/gguf.h"

struct Tensor {
  void* data;
  std::vector<u64> shape;
  enum class Type {
    INVALID,
    F32,
    GGUF_Q8_0,
  } type = Type::INVALID;

  Tensor() = default;
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;
  Tensor(Tensor&& other) noexcept {
    shape = std::move(other.shape);
    type = other.type;
    data = other.data;
    other.data = nullptr;
  }
  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      shape = std::move(other.shape);
      type = other.type;
      data = other.data;
      other.data = nullptr;
    }
    return *this;
  }

  u64 num_elements() const {
    u64 num_elts = 1;
    for (auto dim : shape) {
      num_elts *= dim;
    }
    return num_elts;
  }

  float& ref_f(u64 r, u64 c) {
    assert(type == Type::F32);
    assert(shape.size() == 2);
    // assert(r < shape.at(0));
    assert(c < shape.at(1));
    return ((float*)data)[r * shape.at(1) + c];
  }

  const float& ref_f(u64 r, u64 c) const {
    assert(type == Type::F32);
    assert(shape.size() == 2);
    assert(r < shape.at(0));
    assert(c < shape.at(1));
    return ((const float*)data)[r * shape.at(1) + c];
  }

  float* row_f(u64 r) {
    assert(type == Type::F32);
    assert(shape.size() == 2);
    assert(r < shape.at(0));
    return (float*)data + r * shape.at(1);
  }

  const gguf::Q80Block* row_q80(u64 r) const {
    assert(type == Type::GGUF_Q8_0);
    assert(shape.size() == 2);
    assert(r < shape.at(0));
    assert(shape.at(1) % 32 == 0);
    return (const gguf::Q80Block*)data + r * (shape.at(1) / 32);
  }

  const float* row_f(u64 r) const {
    assert(type == Type::F32);
    assert(shape.size() == 2);
    assert(r < shape.at(0));
    return (const float*)data + r * shape.at(1);
  }

  const float* row_f(u64 m, u64 r) const {
    assert(type == Type::F32);
    assert(shape.size() == 3);
    assert(m < shape.at(0));
    assert(r < shape.at(1));
    const u64 stride = shape.at(1) * shape.at(2);
    return (const float*)data + m * stride + r * shape.at(2);
  }

  float* row_f(u64 m, u64 r) {
    assert(type == Type::F32);
    assert(shape.size() == 3);
    assert(m < shape.at(0));
    assert(r < shape.at(1));
    const u64 stride = shape.at(1) * shape.at(2);
    return (float*)data + m * stride + r * shape.at(2);
  }

  float* as_f32() { return (float*)data; }
  float* as_f32() const { return (float*)data; }

  static Tensor allocate_f32(const std::string& name, const std::vector<u64>& shape) {
    using Store = AlignedStorage<1, 32>;
    Tensor tensor;
    tensor.shape = shape;
    tensor.type = Type::F32;
    u64 num_elements = 1;
    for (auto dim : shape) {
      num_elements *= dim;
    }
    tensor.data = new Store[num_elements * sizeof(float)];
    return tensor;
  }

  void reshape(const std::vector<u64>& new_shape) {
    u64 n = 1;
    for (auto dim : new_shape) {
      n *= dim;
    }
    assert(n == num_elements());
    shape = new_shape;
  }
};
