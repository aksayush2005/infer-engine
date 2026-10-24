#pragma once

#include "llm/tensor.h"
#include "llm/thread_pool.h"

inline float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 x = _mm_add_ps(lo, hi);
  __m128 t = _mm_hadd_ps(x, x);
  __m128 t2 = _mm_hadd_ps(t, t);
  return _mm_cvtss_f32(t2);
}

void matrix_vector_avx2_chunk(Tensor& out,
                              const Tensor& matrix,
                              const Tensor& vector,
                              u64 row_start,
                              u64 row_end);
void matrix_vector_avx2(Tensor& out, const Tensor& matrix, const Tensor& vector);
void matrix_vector_mt(Tensor& out, const Tensor& matrix, const Tensor& vector, ThreadPool& pool);
void extract_row(Tensor& out, const Tensor& in, u64 row);
void extract_rows(Tensor& out, const Tensor& in, const u64* rows, size_t n);
float norm(const Tensor& a);
void scale(Tensor& out, const Tensor& in, float scale);
void copy(Tensor& out, const Tensor& in);
void rms_norm(float* out, const float* in, const float* scale, float epsilon, int n);
void rms_norm(Tensor& out, const Tensor& in, const Tensor& scale, float epsilon);
void rms_norm_batch(Tensor& out, const Tensor& in, const Tensor& scale, float epsilon);
void matrix_matrix_transpose_avx2_unroll_m1_m2(Tensor& out,
                                               const Tensor& matrix1,
                                               const Tensor& matrix2,
                                               u64 start,
                                               u64 end);
void matrix_matrix_transpose(Tensor& out,
                             const Tensor& matrix1,
                             const Tensor& matrix2,
                             ThreadPool& pool);
void set_matrix(Tensor& out, const Tensor& in, u64 i);
void add(Tensor& out, const Tensor& a, const Tensor& b);
void elementwise_multiply(Tensor& out, const Tensor& a);
void set_matrices(Tensor& out, const Tensor& in, u64 i_start);
void gelu_tanh_approx(Tensor& x, u64 size);
bool is_space(char c);
size_t sample_from_logits(float* logits,
                          size_t N,
                          float temperature,
                          std::default_random_engine& rng);
