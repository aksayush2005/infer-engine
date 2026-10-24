#pragma once

#include "llm/tensor.h"

struct QKDims {
  u64 context_length;
  u64 num_q_heads;
  u64 num_kv_heads;
  u64 key_length;
  u64 pos_0;
  u64 pos_1;
};

struct SVDims {
  u64 context_length;
  u64 num_q_heads;
  u64 num_kv_heads;
  u64 value_length;
  u64 pos_0;
  u64 pos_1;
};

float dot(const float* a, const float* b, uint64_t length);
void qk(Tensor& scores, const Tensor& queries, const Tensor& keys, const QKDims& dims);
float exp_approx(float x);
void softmax(float* input, u64 size, u64 stride);
void softmax_cols_inplace(Tensor& input, u64 num_rows);
void sv(Tensor& out, const Tensor& scores, const Tensor& values, const SVDims& dims);
void generate_rope_table(Tensor& table, u64 head_dim, u64 seq_len, float freq_base, float scale);
void apply_rope_table(float* out, const float* in, const float* table_row, u64 n);
void apply_rope_table_rows(Tensor& out,
                           const Tensor& in,
                           const Tensor& table,
                           u64 head_dim,
                           u64 pos,
                           bool increase_pos);
void qk_slice(Tensor& scores,
              const Tensor& queries,
              const Tensor& keys,
              const QKDims& dims,
              u64 q_slice);
void sv_slice(Tensor& out,
              const Tensor& scores,
              const Tensor& values,
              const SVDims& dims,
              u64 out_slice);
