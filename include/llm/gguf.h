#pragma once

#include "llm/common.h"

namespace gguf {

enum ValueType : u32 {
  kU32 = 4,
  kI32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
};

struct Value {
  union {
    u64 uint;
    s64 integer;
    float f32;
    bool boolean;
  };
  std::string string;
  std::vector<Value> array;
  ValueType type;
};

struct Header {
  char magic[4];
  u32 version;
  u64 tensor_count;
  u64 metadata_kv_count;
};

enum class TensorType : u32 {
  kF32 = 0,
  kQ8_0 = 8,
  kQ5 = 13,
};

struct Tensor {
  std::vector<u64> shape;
  TensorType type;
  u64 offset;
};

struct Q80Block {
  static constexpr int kNumValues = 32;
  u16 scale_f16;
  s8 values[kNumValues];
};
static_assert(sizeof(Q80Block) == 34);

void decompress(float* dst, const Q80Block* block);
Value read_typed_value(FileStream* reader, ValueType value_type);
Value read_value(FileStream* reader);
std::unordered_map<std::string, Value> read_metadata(FileStream* reader, u64 n);
std::unordered_map<std::string, Tensor> read_tensors(FileStream* reader, u64 n);

}  // namespace gguf
