#include "llm/gguf.h"

namespace gguf {
void decompress(float* dst, const Q80Block* block) {
  const float s = half_float_to_float(block->scale_f16);
  for (int i = 0; i < Q80Block::kNumValues; i++) {
    dst[i] = s * block->values[i];
  }
}

Value read_typed_value(FileStream* reader, ValueType value_type) {
  switch (value_type) {
    case ValueType::kString: {
      u64 value_len = reader->read<u64>();
      std::string value(value_len, ' ');
      reader->read((void*)value.data(), value_len);
      return {.string = value, .type = value_type};
    } break;
    case ValueType::kU32:
      return {.uint = reader->read<u32>(), .type = value_type};
    case ValueType::kI32:
      return {.integer = reader->read<s32>(), .type = value_type};
    case ValueType::kFloat32:
      return {.f32 = reader->read<float>(), .type = value_type};
    case ValueType::kBool: {
      u8 value = reader->read<u8>();
      assert(value == 0 || value == 1);
      return {.boolean = bool(value), .type = value_type};
    } break;
    case ValueType::kArray: {
      ValueType array_type = reader->read<ValueType>();
      u64 array_len = reader->read<u64>();
      std::vector<Value> array(array_len);
      for (u64 i = 0; i < array_len; i++) {
        array[i] = read_typed_value(reader, array_type);
      }
      return {.array = std::move(array), .type = value_type};
    } break;
    default:
      printf("unknown value type: %d\n", (u32)value_type);
      assert(false);
  }
}

Value read_value(FileStream* reader) {
  ValueType value_type = reader->read<ValueType>();
  return read_typed_value(reader, value_type);
}

std::unordered_map<std::string, Value> read_metadata(FileStream* reader, u64 n) {
  std::unordered_map<std::string, Value> meta;
  for (u64 i = 0; i < n; i++) {
    u64 key_len = reader->read<u64>();
    std::string key(key_len, ' ');
    reader->read((void*)key.data(), key_len);
    auto value = read_value(reader);
    meta.insert({key, value});
  }
  return meta;
}

std::unordered_map<std::string, Tensor> read_tensors(FileStream* reader, u64 n) {
  std::unordered_map<std::string, Tensor> tensors;
  for (u64 i = 0; i < n; i++) {
    u64 name_len = reader->read<u64>();
    std::string name(name_len, ' ');
    reader->read((void*)name.data(), name_len);
    auto& tensor = tensors.insert({name, Tensor{}}).first->second;
    u32 ndim = reader->read<u32>();
    for (u64 j = 0; j < ndim; j++) {
      tensor.shape.push_back(reader->read<u64>());
    }
    tensor.type = static_cast<TensorType>(reader->read<u32>());
    tensor.offset = reader->read<u64>();
  }
  return tensors;
}


}  // namespace gguf
