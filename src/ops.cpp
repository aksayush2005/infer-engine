#include "llm/ops.h"

void matrix_vector_avx2_chunk(Tensor& out,
                              const Tensor& matrix,
                              const Tensor& vector,
                              u64 row_start,
                              u64 row_end) {
  assert(out.shape.size() == 1);
  assert(out.shape.at(0) == matrix.shape.at(0));
  assert(matrix.shape.size() == 2);
  assert(matrix.shape.at(1) == vector.shape.at(0));
  assert(out.type == Tensor::Type::F32);
  assert(matrix.type == Tensor::Type::GGUF_Q8_0);
  assert(vector.type == Tensor::Type::F32);
  assert(row_end <= matrix.shape.at(0));
  assert(row_end > row_start);
  assert(row_end <= matrix.shape.at(0));

  const u64 cols = matrix.shape.at(1);
  float* out_data = (float*)out.data;
  const float* vector_data = (const float*)vector.data;
  const int blocks_per_row = cols / gguf::Q80Block::kNumValues;

  const gguf::Q80Block* block = (const gguf::Q80Block*)matrix.data + (row_start * blocks_per_row);

  assert(cols % gguf::Q80Block::kNumValues == 0);

  for (u64 row = row_start; row < row_end; row++) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    for (int cblock = 0; cblock < blocks_per_row; cblock++) {
      __m128i lo16 = _mm_loadu_si128((const __m128i*)(block->values + 0));
      __m128i hi16 = _mm_loadu_si128((const __m128i*)(block->values + 16));
      __m256i v0_i32 = _mm256_cvtepi8_epi32(lo16);
      __m256i v1_i32 = _mm256_cvtepi8_epi32(_mm_srli_si128(lo16, 8));
      __m256i v2_i32 = _mm256_cvtepi8_epi32(hi16);
      __m256i v3_i32 = _mm256_cvtepi8_epi32(_mm_srli_si128(hi16, 8));
      __m256 v0_f = _mm256_cvtepi32_ps(v0_i32);
      __m256 v1_f = _mm256_cvtepi32_ps(v1_i32);
      __m256 v2_f = _mm256_cvtepi32_ps(v2_i32);
      __m256 v3_f = _mm256_cvtepi32_ps(v3_i32);
      __m256 scale = _mm256_set1_ps(half_float_to_float(block->scale_f16));
      v0_f = _mm256_mul_ps(v0_f, scale);
      v1_f = _mm256_mul_ps(v1_f, scale);
      v2_f = _mm256_mul_ps(v2_f, scale);
      v3_f = _mm256_mul_ps(v3_f, scale);
      __m256 vector0 = _mm256_load_ps(vector_data + cblock * gguf::Q80Block::kNumValues);
      __m256 vector1 = _mm256_load_ps(vector_data + cblock * gguf::Q80Block::kNumValues + 8);
      __m256 vector2 = _mm256_load_ps(vector_data + cblock * gguf::Q80Block::kNumValues + 16);
      __m256 vector3 = _mm256_load_ps(vector_data + cblock * gguf::Q80Block::kNumValues + 24);
      sum0 = _mm256_fmadd_ps(vector0, v0_f, sum0);
      sum1 = _mm256_fmadd_ps(vector1, v1_f, sum1);
      sum2 = _mm256_fmadd_ps(vector2, v2_f, sum2);
      sum3 = _mm256_fmadd_ps(vector3, v3_f, sum3);
      block++;
    }
    out_data[row] = hsum(_mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)));
  }
}

void matrix_vector_avx2(Tensor& out, const Tensor& matrix, const Tensor& vector) {
  matrix_vector_avx2_chunk(out, matrix, vector, 0, matrix.shape.at(0));
}

void matrix_vector_mt(Tensor& out, const Tensor& matrix, const Tensor& vector, ThreadPool& pool) {
  const u64 rows = matrix.shape.at(0);
  const u64 rows_per_thread = rows / pool.num_threads();
  std::atomic<u64> pending = pool.num_threads();
  for (u64 i = 0; i < pool.num_threads(); i++) {
    u64 start_row = i * rows_per_thread;
    u64 end_row = (i + 1) * rows_per_thread;
    if (i == pool.num_threads() - 1) {
      end_row = rows;
    }
    pool.enqueue([&, start_row, end_row]() {
      matrix_vector_avx2_chunk(out, matrix, vector, start_row, end_row);
      pending--;
    });
  }
  pool.wait();
  assert(pending == 0);
}

void extract_row(Tensor& out, const Tensor& in, u64 row) {
  assert(out.shape.size() == 1);
  assert(out.shape.at(0) == in.shape.at(1));
  assert(in.shape.size() == 2);
  assert(out.type == Tensor::Type::F32);
  assert(row < in.shape.at(0));
  const u64 cols = in.shape.at(1);
  switch (in.type) {
    case Tensor::Type::F32:
      memcpy(out.data, in.row_f(row), cols * sizeof(float));
      break;
    case Tensor::Type::GGUF_Q8_0: {
      assert(cols % gguf::Q80Block::kNumValues == 0);
      const u64 blocks_per_row = cols / gguf::Q80Block::kNumValues;
      const u64 start_block = row * blocks_per_row;
      const gguf::Q80Block* block = (const gguf::Q80Block*)in.data + start_block;
      float* dest = (float*)out.data;
      for (u64 i = 0; i < blocks_per_row; i++) {
        decompress(dest, block);
        block++;
        dest += gguf::Q80Block::kNumValues;
      }
    } break;
    default:
      assert(false);
  }
}

void extract_rows(Tensor& out, const Tensor& in, const u64* rows, size_t n) {
  assert(out.shape.size() == 2);
  assert(out.shape.at(0) == n);
  assert(out.shape.at(1) == in.shape.at(1));
  assert(in.shape.size() == 2);
  assert(out.type == Tensor::Type::F32);
  assert(in.type == Tensor::Type::GGUF_Q8_0);
  const u64 cols = in.shape.at(1);

  float* dest = (float*)out.data;
  for (size_t i = 0; i < n; i++) {
    u64 row = rows[i];
    assert(row < in.shape.at(0));
    const u64 blocks_per_row = cols / gguf::Q80Block::kNumValues;
    const u64 start_block = row * blocks_per_row;
    const gguf::Q80Block* block = (const gguf::Q80Block*)in.data + start_block;
    for (u64 i = 0; i < blocks_per_row; i++) {
      decompress(dest, block);
      block++;
      dest += gguf::Q80Block::kNumValues;
    }
  }
}

float norm(const Tensor& a) {
  assert(a.shape.size() == 1);
  assert(a.type == Tensor::Type::F32);
  float sum = 0.0f;
  const float* a_data = (const float*)a.data;
  for (u64 i = 0; i < a.shape.at(0); i++) {
    sum += a_data[i] * a_data[i];
  }
  return std::sqrt(sum);
}

void scale(Tensor& out, const Tensor& in, float scale) {
  const u64 n = out.num_elements();
  assert(out.type == Tensor::Type::F32);
  assert(in.type == Tensor::Type::F32);
  assert(in.shape == out.shape);
  // ok for out/in to alias.
  float* out_data = (float*)out.data;
  const float* in_data = (const float*)in.data;
  for (u64 i = 0; i < n; i++) {
    out_data[i] = in_data[i] * scale;
  }
}

void copy(Tensor& out, const Tensor& in) {
  assert(out.shape == in.shape);
  assert(out.type == Tensor::Type::F32);
  assert(in.type == Tensor::Type::F32);
  float* out_data = (float*)out.data;
  const float* in_data = (const float*)in.data;
  const u64 n = out.num_elements();
  for (u64 i = 0; i < n; i++) {
    out_data[i] = in_data[i];
  }
}

void rms_norm(float* out, const float* in, const float* scale, float epsilon, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += in[i] * in[i];
  }
  float rms = std::sqrt(epsilon + sum / n);
  const float inv_rms = 1.0f / rms;
  for (int i = 0; i < n; i++) {
    // seems like they add 1 already?
    out[i] = in[i] * scale[i] * inv_rms;
  }
}

void rms_norm(Tensor& out, const Tensor& in, const Tensor& scale, float epsilon) {
  assert(out.shape.size() == 1);
  assert(out.shape.at(0) == in.shape.at(0));
  assert(out.type == Tensor::Type::F32);
  assert(in.type == Tensor::Type::F32);
  assert(scale.shape.size() == 1);
  assert(scale.shape.at(0) == in.shape.at(0));
  assert(scale.type == Tensor::Type::F32);
  const u64 n = in.shape.at(0);
  float* out_data = (float*)out.data;
  const float* in_data = (const float*)in.data;
  const float* scale_data = (const float*)scale.data;
  rms_norm(out_data, in_data, scale_data, epsilon, n);
}

void rms_norm_batch(Tensor& out, const Tensor& in, const Tensor& scale, float epsilon) {
  assert(out.type == Tensor::Type::F32);
  assert(in.type == Tensor::Type::F32);
  assert(scale.type == Tensor::Type::F32);
  assert(scale.shape.size() == 1);
  assert(out.shape == in.shape);
  assert(out.shape.size() > 0);
  assert(out.shape.back() == scale.shape[0]);
  const u64 cols = out.shape.back();

  float* out_data = (float*)out.data;
  const float* in_data = (const float*)in.data;
  const float* scale_data = (const float*)scale.data;

  u64 n_elem = out.num_elements();
  u64 n_iters = n_elem / cols;
  assert(n_elem % cols == 0);

  for (u64 i = 0; i < n_iters; i++) {
    rms_norm(out_data, in_data, scale_data, epsilon, cols);
    in_data += cols;
    out_data += cols;
  }
}

void matrix_matrix_transpose_avx2_unroll_m1_m2(Tensor& out,
                                               const Tensor& matrix1,
                                               const Tensor& matrix2,
                                               u64 start,
                                               u64 end) {
  assert(out.shape.size() == 2);
  assert(matrix1.shape.size() == 2);
  assert(matrix2.shape.size() == 2);

  const u64 n = matrix1.shape.at(1);
  assert(n % gguf::Q80Block::kNumValues == 0);
  const int num_blocks = n / gguf::Q80Block::kNumValues;
  assert(matrix2.shape.at(1) == n);

  const u64 outer2 = out.shape.at(0);
  assert(outer2 == matrix2.shape.at(0));

  const u64 outer1 = out.shape.at(1);
  assert(outer1 == matrix1.shape.at(0));
  assert(start < outer1);
  assert(end <= outer1);
  assert(start < end);

  for (u64 m1r = start; m1r < end; m1r += 2) {
    const gguf::Q80Block* m1_row_data = matrix1.row_q80(m1r);
    for (u64 m2r = 0; m2r < outer2; m2r += 2) {
      const float* m2_row_data = matrix2.row_f(m2r);
      const gguf::Q80Block* block0 = m1_row_data;
      const gguf::Q80Block* block1 = m1_row_data + num_blocks;

      __m256 sum0_0 = _mm256_setzero_ps();  // m1, m2 indexing
      __m256 sum0_1 = _mm256_setzero_ps();
      __m256 sum1_0 = _mm256_setzero_ps();
      __m256 sum1_1 = _mm256_setzero_ps();

      for (int cblock = 0; cblock < num_blocks; cblock++) {
        __m256 scale0 = _mm256_set1_ps(half_float_to_float(block0->scale_f16));
        __m256 scale1 = _mm256_set1_ps(half_float_to_float(block1->scale_f16));

        u64 block_offset = cblock * gguf::Q80Block::kNumValues;

        __m128i lo16_0 = _mm_loadu_si128((const __m128i*)(block0->values + 0));
        __m128i lo16_1 = _mm_loadu_si128((const __m128i*)(block1->values + 0));
        {
          auto values0_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16_0));
          values0_0 = _mm256_mul_ps(values0_0, scale0);
          auto values1_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo16_1));
          values1_0 = _mm256_mul_ps(values1_0, scale1);
          __m256 vector0_0 = _mm256_load_ps(m2_row_data + block_offset);
          __m256 vector1_0 = _mm256_load_ps(m2_row_data + block_offset + n);
          sum0_0 = _mm256_fmadd_ps(vector0_0, values0_0, sum0_0);
          sum0_1 = _mm256_fmadd_ps(vector1_0, values0_0, sum0_1);
          sum1_0 = _mm256_fmadd_ps(vector0_0, values1_0, sum1_0);
          sum1_1 = _mm256_fmadd_ps(vector1_0, values1_0, sum1_1);
        }

        {
          auto values0_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo16_0, 8)));
          values0_1 = _mm256_mul_ps(values0_1, scale0);
          auto values1_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo16_1, 8)));
          values1_1 = _mm256_mul_ps(values1_1, scale1);
          __m256 vector0_1 = _mm256_load_ps(m2_row_data + block_offset + 8);
          __m256 vector1_1 = _mm256_load_ps(m2_row_data + block_offset + 8 + n);
          sum0_0 = _mm256_fmadd_ps(vector0_1, values0_1, sum0_0);
          sum0_1 = _mm256_fmadd_ps(vector1_1, values0_1, sum0_1);
          sum1_0 = _mm256_fmadd_ps(vector0_1, values1_1, sum1_0);
          sum1_1 = _mm256_fmadd_ps(vector1_1, values1_1, sum1_1);
        }

        __m128i hi16_0 = _mm_loadu_si128((const __m128i*)(block0->values + 16));
        __m128i hi16_1 = _mm_loadu_si128((const __m128i*)(block1->values + 16));
        {
          auto values0_2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16_0));
          values0_2 = _mm256_mul_ps(values0_2, scale0);
          auto values1_2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi16_1));
          values1_2 = _mm256_mul_ps(values1_2, scale1);
          __m256 vector0_2 = _mm256_load_ps(m2_row_data + block_offset + 16);
          __m256 vector1_2 = _mm256_load_ps(m2_row_data + block_offset + 16 + n);
          sum0_0 = _mm256_fmadd_ps(vector0_2, values0_2, sum0_0);
          sum0_1 = _mm256_fmadd_ps(vector1_2, values0_2, sum0_1);
          sum1_0 = _mm256_fmadd_ps(vector0_2, values1_2, sum1_0);
          sum1_1 = _mm256_fmadd_ps(vector1_2, values1_2, sum1_1);
        }

        {
          auto values0_3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi16_0, 8)));
          values0_3 = _mm256_mul_ps(values0_3, scale0);
          auto values1_3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi16_1, 8)));
          values1_3 = _mm256_mul_ps(values1_3, scale1);
          __m256 vector0_3 = _mm256_load_ps(m2_row_data + block_offset + 24);
          __m256 vector1_3 = _mm256_load_ps(m2_row_data + block_offset + 24 + n);
          sum0_0 = _mm256_fmadd_ps(vector0_3, values0_3, sum0_0);
          sum0_1 = _mm256_fmadd_ps(vector1_3, values0_3, sum0_1);
          sum1_0 = _mm256_fmadd_ps(vector0_3, values1_3, sum1_0);
          sum1_1 = _mm256_fmadd_ps(vector1_3, values1_3, sum1_1);
        }
        block1++;
        block0++;
      }
      out.ref_f(m2r, m1r) = hsum(sum0_0);
      out.ref_f(m2r + 1, m1r) = hsum(sum0_1);
      out.ref_f(m2r, m1r + 1) = hsum(sum1_0);
      out.ref_f(m2r + 1, m1r + 1) = hsum(sum1_1);
    }
  }
}

void matrix_matrix_transpose(Tensor& out,
                             const Tensor& matrix1,
                             const Tensor& matrix2,
                             ThreadPool& pool) {
  const u64 rows = matrix1.shape.at(0);
  const u64 rows_per_thread = rows / pool.num_threads();
  std::atomic<u64> pending = pool.num_threads();

  for (size_t i = 0; i < pool.num_threads(); i++) {
    u64 start_row = i * rows_per_thread;
    u64 end_row = (i + 1) * rows_per_thread;
    if (i == pool.num_threads() - 1) {
      end_row = rows;
    }
    pool.enqueue([&, start_row, end_row]() {
      if ((end_row - start_row) % 2) {
        printf("unsupported number of rows: %d for %d threads\n", (int)(end_row - start_row),
               (int)pool.num_threads());
        exit(-1);
      } else {
        matrix_matrix_transpose_avx2_unroll_m1_m2(out, matrix1, matrix2, start_row, end_row);
      }
      pending--;
    });
  }
  pool.wait();

  assert(pending == 0);
}

void set_matrix(Tensor& out, const Tensor& in, u64 i) {
  assert(in.type == Tensor::Type::F32);
  assert(out.type == Tensor::Type::F32);
  assert(out.shape.size() == 3);
  assert(out.shape.at(1) == in.shape.at(0));
  assert(out.shape.at(2) == in.shape.at(1));
  assert(i < out.shape.at(0));

  const u64 stride = out.shape.at(1) * out.shape.at(2);
  const float* in_data = (const float*)in.data;
  float* out_data = (float*)out.data + stride * i;
  memcpy(out_data, in_data, stride * sizeof(float));
}

void add(Tensor& out, const Tensor& a, const Tensor& b) {
  assert(out.type == Tensor::Type::F32);
  assert(a.type == Tensor::Type::F32);
  assert(b.type == Tensor::Type::F32);
  assert(out.shape == a.shape);
  assert(out.shape == b.shape);

  const u64 n = out.num_elements();
  float* out_data = (float*)out.data;
  const float* a_data = (const float*)a.data;
  const float* b_data = (const float*)b.data;
  for (u64 i = 0; i < n; i++) {
    out_data[i] = a_data[i] + b_data[i];
  }
}

void elementwise_multiply(Tensor& out, const Tensor& a) {
  assert(out.type == Tensor::Type::F32);
  assert(a.type == Tensor::Type::F32);
  assert(out.shape == a.shape);
  const u64 n = out.num_elements();
  float* out_data = (float*)out.data;
  const float* a_data = (const float*)a.data;
  for (u64 i = 0; i < n; i++) {
    out_data[i] *= a_data[i];
  }
}

void set_matrices(Tensor& out, const Tensor& in, u64 i_start) {
  assert(in.type == Tensor::Type::F32);
  assert(out.type == Tensor::Type::F32);

  assert(out.shape.size() == 3);
  assert(in.shape.size() == 3);
  assert(out.shape.at(1) == in.shape.at(1));
  assert(out.shape.at(2) == in.shape.at(2));

  for (u64 iter = 0; iter < in.shape.at(0); iter++) {
    u64 i = i_start + iter;
    assert(i < out.shape.at(0));
    const u64 stride = out.shape.at(1) * out.shape.at(2);
    const float* in_data = (const float*)in.data + stride * iter;
    float* out_data = (float*)out.data + stride * i;
    memcpy(out_data, in_data, stride * sizeof(float));
  }
}

void gelu_tanh_approx(Tensor& x, u64 size) {
  assert(x.type == Tensor::Type::F32);
  assert(x.num_elements() == size);
  float* x_f = (float*)x.data;

  constexpr float c1 = 0.7978845608;
  constexpr float c2 = 0.044715f;
  for (size_t i = 0; i < size; i++) {
    x_f[i] = 0.5f * x_f[i] * (1.f + std::tanh(c1 * (x_f[i] + c2 * x_f[i] * x_f[i] * x_f[i])));
  }
}

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

size_t sample_from_logits(float* logits,
                          size_t N,
                          float temperature,
                          std::default_random_engine& rng) {
  float max_logit = logits[0];
  for (size_t i = 1; i < N; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  std::vector<float> probs(N);
  float sum = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    probs[i] = std::exp((logits[i] - max_logit) / temperature);
    sum += probs[i];
  }

  for (size_t i = 0; i < N; ++i) {
    probs[i] /= sum;
  }

  std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
  return dist(rng);
}

