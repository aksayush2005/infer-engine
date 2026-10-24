#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <xmmintrin.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// how much to shrink the maximum context length by.
// this reduces memory usage.
constexpr uint kContextDivisor = 16;
// number of threads to use for parallel computation.
constexpr uint kNumThreads = 16;
// number of tokens to process in a single batch.
constexpr uint kBatchSize = 32;
// temperature for sampling. 1 is recommended.
constexpr float kTemperature = 1.0f;

constexpr bool kShowStats = true;

constexpr const char* kSystemPrompt = "";

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;

template <u64 size_bytes, u64 alignment>
struct AlignedStorage {
  alignas(alignment) u8 data[size_bytes];
};

inline float as_float(u32 x) {
  float result;
  memcpy(&result, &x, sizeof(float));
  return result;
}

inline float half_float_to_float(u16 x) {
  const u32 sign_bit = x & 0x8000;
  u32 result = x;              // start with the f16
  result &= 0x7fff;            // remove sign bit
  result <<= 13;               // realign to f32 layout
  result += 0x38000000;        // adjust exponent
  result |= (sign_bit << 16);  // add back sign bit.
  return as_float(result);
}

struct FileStream {
  FileStream(const void* x) : data(x) {}
  const void* data;
  size_t pos = 0;

  template <typename T>
  T read() {
    T value;
    memcpy(&value, (const char*)data + pos, sizeof(T));
    pos += sizeof(T);
    return value;
  }

  void read(void* ptr, size_t size) {
    memcpy(ptr, (const char*)data + pos, size);
    pos += size;
  }
};
