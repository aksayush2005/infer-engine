#include "llm/attention.h"
#include "llm/ops.h"

float dot(const float* a, const float* b, uint64_t length) {
  assert((length % 32) == 0);
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  for (uint64_t i = 0; i < length; i += 32) {
    __m256 a0 = _mm256_load_ps(a + i + 0);
    __m256 a1 = _mm256_load_ps(a + i + 8);
    __m256 a2 = _mm256_load_ps(a + i + 16);
    __m256 a3 = _mm256_load_ps(a + i + 24);
    __m256 b0 = _mm256_load_ps(b + i + 0);
    __m256 b1 = _mm256_load_ps(b + i + 8);
    __m256 b2 = _mm256_load_ps(b + i + 16);
    __m256 b3 = _mm256_load_ps(b + i + 24);
    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
    acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    acc2 = _mm256_fmadd_ps(a2, b2, acc2);
    acc3 = _mm256_fmadd_ps(a3, b3, acc3);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  acc2 = _mm256_add_ps(acc2, acc3);
  acc0 = _mm256_add_ps(acc0, acc2);
  return hsum(acc0);
}

void qk(Tensor& scores, const Tensor& queries, const Tensor& keys, const QKDims& dims) {
  assert(scores.type == Tensor::Type::F32);
  assert(queries.type == Tensor::Type::F32);
  assert(keys.type == Tensor::Type::F32);
  // scores is context_length, num_q_heads
  assert(scores.shape.size() == 2);
  assert(dims.context_length == scores.shape.at(0));
  assert(dims.num_q_heads == scores.shape.at(1));
  // queries is num_q_heads, key_length
  assert(queries.shape.size() == 2);
  assert(dims.num_q_heads == queries.shape.at(0));
  assert(dims.key_length == queries.shape.at(1));
  // keys is context_length, num_kv_heads, key_length
  assert(keys.shape.size() == 3);
  assert(dims.context_length == keys.shape.at(0));
  assert(dims.num_kv_heads == keys.shape.at(1));
  assert(dims.key_length == keys.shape.at(2));
  assert(dims.pos_0 < dims.context_length);
  assert(dims.pos_1 <= dims.context_length);
  assert(dims.pos_1 > dims.pos_0);
  assert(dims.num_q_heads % dims.num_kv_heads == 0);
  const u64 qs_per_kv = dims.num_q_heads / dims.num_kv_heads;

  for (u64 pos = dims.pos_0; pos < dims.pos_1; pos++) {
    for (u64 qhead = 0; qhead < dims.num_q_heads; qhead++) {
      u64 kvhead = qhead / qs_per_kv;
      const float* query = queries.row_f(qhead);
      const float* key = keys.row_f(pos, kvhead);
      scores.ref_f(pos - dims.pos_0, qhead) = dot(query, key, dims.key_length);
    }
  }
}

float exp_approx(float x) {
  // todo vectorize this :)
  float a = (1 << 22) / float(M_LN2);
  s32 b = 127 * (1 << 23);
  s32 r = a * x;
  s32 s = b + r;
  s32 t = b - r;
  return as_float(s) / as_float(t);
}

inline float tanh_approx(float x) {
  return std::tanh(x);
}

void softmax(float* input, u64 size, u64 stride) {
  float max_val = input[0];
  for (u64 i = 1; i < size; i++) {
    if (input[i * stride] > max_val) {
      max_val = input[i * stride];
    }
  }

  float sum_exp = 0.0;
  for (u64 i = 0; i < size; i++) {
    // input[i * stride] = std::exp(input[i * stride] - max_val);
    input[i * stride] = exp_approx(input[i * stride] - max_val);
    sum_exp += input[i * stride];
  }

  for (u64 i = 0; i < size; i++) {
    input[i * stride] /= sum_exp;
  }
}

void softmax_cols_inplace(Tensor& input, u64 num_rows) {
  assert(input.shape.size() == 2);
  assert(input.type == Tensor::Type::F32);
  assert(num_rows <= input.shape.at(0));
  const u64 cols = input.shape.at(1);
  for (u64 col = 0; col < cols; col++) {
    softmax(&input.ref_f(0, col), num_rows, cols);
  }
}

void sv(Tensor& out, const Tensor& scores, const Tensor& values, const SVDims& dims) {
  assert(out.type == Tensor::Type::F32);
  assert(scores.type == Tensor::Type::F32);
  assert(values.type == Tensor::Type::F32);
  // out is [num_q_heads, value_length]
  assert(out.shape.size() == 2);
  assert(dims.num_q_heads == out.shape.at(0));
  assert(dims.value_length == out.shape.at(1));
  // scores is [context_length, num_q_heads]
  assert(scores.shape.size() == 2);
  assert(dims.context_length == scores.shape.at(0));
  assert(dims.num_q_heads == scores.shape.at(1));
  // values is [context_length, num_kv_heads, value_length]
  assert(values.shape.size() == 3);
  assert(dims.context_length == values.shape.at(0));
  assert(dims.num_kv_heads == values.shape.at(1));
  assert(dims.value_length == values.shape.at(2));
  assert(dims.num_q_heads % dims.num_kv_heads == 0);
  const u64 qs_per_kv = dims.num_q_heads / dims.num_kv_heads;

  // set out to 0
  float* out_data = (float*)out.data;
  for (u64 i = 0; i < dims.num_q_heads * dims.value_length; i++) {
    out_data[i] = 0.0f;
  }

  for (u64 cpos = dims.pos_0; cpos < dims.pos_1; cpos++) {
    const u64 spos = cpos - dims.pos_0;

    for (u64 qhead = 0; qhead < dims.num_q_heads; qhead++) {
      const u64 kvhead = qhead / qs_per_kv;
      const float score = scores.ref_f(spos, qhead);
      const float* values_row = values.row_f(cpos, kvhead);
      float* out_row = out.row_f(qhead);

      for (u64 i = 0; i < dims.value_length; i++) {
        out_row[i] += score * values_row[i];
      }
    }
  }
}

void generate_rope_table(Tensor& table, u64 head_dim, u64 seq_len, float freq_base, float scale) {
  assert(table.shape.size() == 2);
  assert(table.shape.at(0) == seq_len * 2);
  assert(table.shape.at(1) == head_dim);
  assert(table.type == Tensor::Type::F32);
  for (u64 i = 0; i < head_dim / 2; i++) {
    const double freq = powf(freq_base, -2 * float(i) / float(head_dim)) / scale;
    for (u64 pos = 0; pos < seq_len * 2; pos++) {
      const double angle = freq * pos;
      table.ref_f(pos, i * 2) = std::sin(angle);
      table.ref_f(pos, i * 2 + 1) = std::cos(angle);
    }
  }
}

void apply_rope_table(float* out, const float* in, const float* table_row, u64 n) {
  const float* in_1 = in + n / 2;
  float* out_1 = out + n / 2;
  for (u64 i = 0; i < n / 2; i++) {
    const float x_in = in[i];
    const float y_in = in_1[i];
    const float s = table_row[i * 2];
    const float c = table_row[i * 2 + 1];
    out[i] = x_in * c - y_in * s;
    out_1[i] = x_in * s + y_in * c;
  }
}

void apply_rope_table_rows(Tensor& out,
                           const Tensor& in,
                           const Tensor& table,
                           u64 head_dim,
                           u64 pos,
                           bool increase_pos) {
  // check out
  assert(out.shape.size() > 0);
  assert(out.shape.back() == head_dim);
  assert(out.type == Tensor::Type::F32);

  // check in
  assert(in.shape.size() > 0);
  assert(in.shape.back() == head_dim);
  assert(in.type == Tensor::Type::F32);

  assert(out.shape == in.shape);
  // const u64 rows = in.shape.at(0);
  assert(in.num_elements() % head_dim == 0);
  // u64 n_iters = in.num_elements() / head_dim;
  u64 n_inner_iters = in.num_elements() / head_dim;
  u64 n_outer_iters = 1;

  if (increase_pos) {
    n_outer_iters = in.shape.at(0);
    n_inner_iters /= n_outer_iters;
    assert(n_inner_iters * n_outer_iters * head_dim == in.num_elements());
  }

  // check table
  assert(table.shape.size() == 2);
  assert(table.shape.at(0) > pos);
  assert(table.shape.at(1) == head_dim);
  assert(table.type == Tensor::Type::F32);

  const float* in_row = (const float*)in.data;
  float* out_row = (float*)out.data;
  for (u64 o = 0; o < n_outer_iters; o++) {
    const float* table_row = table.row_f(pos);

    for (u64 i = 0; i < n_inner_iters; i++) {
      apply_rope_table(out_row, in_row, table_row, head_dim);
      in_row += head_dim;
      out_row += head_dim;
    }
    pos++;
  }
}

void qk_slice(Tensor& scores,
              const Tensor& queries,
              const Tensor& keys,
              const QKDims& dims,
              u64 q_slice) {
  assert(scores.type == Tensor::Type::F32);
  assert(queries.type == Tensor::Type::F32);
  assert(keys.type == Tensor::Type::F32);

  // scores is context_length, num_q_heads
  assert(scores.shape.size() == 2);
  assert(dims.context_length == scores.shape.at(0));
  assert(dims.num_q_heads == scores.shape.at(1));

  // queries is num_q_heads, key_length
  assert(queries.shape.size() == 3);
  assert(dims.num_q_heads == queries.shape.at(1));
  assert(dims.key_length == queries.shape.at(2));
  assert(q_slice < queries.shape.at(0));

  // keys is context_length, num_kv_heads, key_length
  assert(keys.shape.size() == 3);
  assert(dims.context_length == keys.shape.at(0));
  assert(dims.num_kv_heads == keys.shape.at(1));
  assert(dims.key_length == keys.shape.at(2));

  assert(dims.pos_0 < dims.context_length);
  assert(dims.pos_1 <= dims.context_length);
  assert(dims.pos_1 > dims.pos_0);

  assert(dims.num_q_heads % dims.num_kv_heads == 0);
  const u64 qs_per_kv = dims.num_q_heads / dims.num_kv_heads;

  for (u64 pos = dims.pos_0; pos < dims.pos_1; pos++) {
    for (u64 qhead = 0; qhead < dims.num_q_heads; qhead++) {
      u64 kvhead = qhead / qs_per_kv;
      const float* query = queries.row_f(q_slice, qhead);
      const float* key = keys.row_f(pos, kvhead);
      scores.ref_f(pos - dims.pos_0, qhead) = dot(query, key, dims.key_length);
    }
  }
}

void sv_slice(Tensor& out,
              const Tensor& scores,
              const Tensor& values,
              const SVDims& dims,
              u64 out_slice) {
  assert(out.type == Tensor::Type::F32);
  assert(scores.type == Tensor::Type::F32);
  assert(values.type == Tensor::Type::F32);

  // out is [num_q_heads, value_length]
  assert(out.shape.size() == 3);
  assert(dims.num_q_heads == out.shape.at(1));
  assert(dims.value_length == out.shape.at(2));
  assert(out_slice < out.shape.at(0));

  // scores is [context_length, num_q_heads]
  assert(scores.shape.size() == 2);
  assert(dims.context_length == scores.shape.at(0));
  assert(dims.num_q_heads == scores.shape.at(1));

  // values is [context_length, num_kv_heads, value_length]
  assert(values.shape.size() == 3);
  assert(dims.context_length == values.shape.at(0));
  assert(dims.num_kv_heads == values.shape.at(1));
  assert(dims.value_length == values.shape.at(2));

  assert(dims.num_q_heads % dims.num_kv_heads == 0);
  const u64 qs_per_kv = dims.num_q_heads / dims.num_kv_heads;

  // set out to 0
  float* out_data = out.row_f(out_slice, 0);
  for (u64 i = 0; i < dims.num_q_heads * dims.value_length; i++) {
    out_data[i] = 0.0f;
  }

  for (u64 cpos = dims.pos_0; cpos < dims.pos_1; cpos++) {
    const u64 spos = cpos - dims.pos_0;

    for (u64 qhead = 0; qhead < dims.num_q_heads; qhead++) {
      const u64 kvhead = qhead / qs_per_kv;
      const float score = scores.ref_f(spos, qhead);
      const float* values_row = values.row_f(cpos, kvhead);
      float* out_row = out.row_f(out_slice, qhead);

      for (u64 i = 0; i < dims.value_length; i++) {
        out_row[i] += score * values_row[i];
      }
    }
  }
}
