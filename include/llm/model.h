#pragma once

#include "llm/gguf.h"
#include "llm/tensor.h"
#include "llm/thread_pool.h"
#include "llm/ops.h"
#include "llm/attention.h"

class Model {
 public:
  Model(const u8* tensor_data,
        const std::unordered_map<std::string, gguf::Tensor>& tensors,
        const std::unordered_map<std::string, gguf::Value>& metadata)
      : m_tensor_data(tensor_data), m_tensors(tensors), m_thread_pool(kNumThreads), m_rng(111) {
    // Read metadata
    m_dim = metadata.at("gemma3.embedding_length").integer;
    m_num_q_heads = metadata.at("gemma3.attention.head_count").integer;
    m_num_kv_heads = metadata.at("gemma3.attention.head_count_kv").integer;
    m_key_length = metadata.at("gemma3.attention.key_length").integer;
    m_value_length = metadata.at("gemma3.attention.value_length").integer;
    m_rms_epsilon = metadata.at("gemma3.attention.layer_norm_rms_epsilon").f32;
    m_context_length = metadata.at("gemma3.context_length").integer;
    m_ff_length = metadata.at("gemma3.feed_forward_length").integer;
    m_sliding_window_size = metadata.at("gemma3.attention.sliding_window").integer;
    // unclear which norms use hard-coded 1e-6 vs the config param
    // so we only continue if the config param is the same as the hard-coded value
    assert(m_rms_epsilon == 1e-6f);

    m_vocab_size = tensors.at("token_embd.weight").shape.at(1);
    // decrease context length so rope tables generate faster.
    m_max_ctxt = m_context_length / kContextDivisor;

    using gguf::TensorType;
    m_layer_xin = allocate_f32("x_in", {m_dim});
    m_local_rope_table = allocate_f32("local_rope_table", {m_max_ctxt * 2, m_key_length});
    m_global_rope_table = allocate_f32("global_rope_table", {m_max_ctxt * 2, m_key_length});

    printf("Generating rope tables for context length %ld...\n", m_max_ctxt);
    m_thread_pool.enqueue([&]() {
      generate_rope_table(m_local_rope_table, m_key_length, m_max_ctxt, 10000.0f, 1.f);
    });
    m_thread_pool.enqueue([&]() {
      generate_rope_table(m_global_rope_table, m_key_length, m_max_ctxt, 1000000.0f, 1.f);
    });
    m_thread_pool.wait();
    m_logits = allocate_f32("logits", {m_vocab_size});
    m_embedding = load_tensor("token_embd.weight", {m_vocab_size, m_dim}, TensorType::kQ8_0);
    m_output_norm = load_tensor("output_norm.weight", {m_dim}, TensorType::kF32);

    {
      auto& tc = m_tcache;
      tc.xskip = allocate_f32("x_skip", {m_dim});
      tc.xattn = allocate_f32("x_attn", {m_dim});
      tc.xff = allocate_f32("x_ff", {m_dim});
      tc.queries = allocate_f32("queries", {m_num_q_heads, m_key_length});
      tc.keys = allocate_f32("keys", {m_num_kv_heads, m_key_length});
      tc.values = allocate_f32("values", {m_num_kv_heads, m_value_length});
      tc.scores = allocate_f32("scores", {m_max_ctxt, m_num_q_heads});
      tc.context = allocate_f32("context", {m_num_q_heads, m_value_length});
      tc.attn_result = allocate_f32("attn_result", {m_dim});
      tc.up = allocate_f32("up", {m_ff_length});
      tc.gate = allocate_f32("gate", {m_ff_length});
    }

    {
      m_b.x_in = allocate_f32("x_in", {kBatchSize, m_dim});
      m_b.skip = allocate_f32("skip", {kBatchSize, m_dim});
      m_b.keys = allocate_f32("keys", {kBatchSize, m_num_kv_heads, m_key_length});
      m_b.values = allocate_f32("values", {kBatchSize, m_num_kv_heads, m_value_length});
      m_b.queries = allocate_f32("queries", {kBatchSize, m_num_q_heads, m_key_length});
      m_b.context = allocate_f32("context", {kBatchSize, m_num_q_heads, m_value_length});
      m_b.attn_result = allocate_f32("attn_result", {kBatchSize, m_dim});
      m_b.up = allocate_f32("up", {kBatchSize, m_ff_length});
      m_b.gate = allocate_f32("gate", {kBatchSize, m_ff_length});
      m_b.xff = allocate_f32("xff", {kBatchSize, m_dim});
    }

    m_layers.resize(metadata.at("gemma3.block_count").integer);
    for (u64 i = 0; i < m_layers.size(); i++) {
      auto& layer = m_layers[i];
      std::string prefix = std::string("blk.") + std::to_string(i).c_str();
      // 5 local layers for every global layer, starting with a local layer.
      layer.local = (i % 6) != 5;
      layer.attn_norm = load_tensor(prefix + ".attn_norm.weight", {m_dim}, TensorType::kF32);
      layer.norm_q = load_tensor(prefix + ".attn_q_norm.weight", {m_key_length}, TensorType::kF32);
      layer.norm_k = load_tensor(prefix + ".attn_k_norm.weight", {m_key_length}, TensorType::kF32);
      layer.ff_norm = load_tensor(prefix + ".ffn_norm.weight", {m_dim}, TensorType::kF32);
      layer.post_ff_norm = load_tensor(prefix + ".post_ffw_norm.weight", {m_dim}, TensorType::kF32);
      layer.post_attn_norm =
          load_tensor(prefix + ".post_attention_norm.weight", {m_dim}, TensorType::kF32);
      layer.wk = load_tensor(prefix + ".attn_k.weight", {m_num_kv_heads * m_key_length, m_dim},
                             TensorType::kQ8_0);
      layer.wq = load_tensor(prefix + ".attn_q.weight", {m_num_q_heads * m_key_length, m_dim},
                             TensorType::kQ8_0);
      layer.wv = load_tensor(prefix + ".attn_v.weight", {m_num_kv_heads * m_value_length, m_dim},
                             TensorType::kQ8_0);
      layer.attn_output = load_tensor(prefix + ".attn_output.weight",
                                      {m_dim, m_num_q_heads * m_value_length}, TensorType::kQ8_0);
      layer.ff_up = load_tensor(prefix + ".ffn_up.weight", {m_ff_length, m_dim}, TensorType::kQ8_0);
      layer.ff_gate =
          load_tensor(prefix + ".ffn_gate.weight", {m_ff_length, m_dim}, TensorType::kQ8_0);
      layer.ff_down =
          load_tensor(prefix + ".ffn_down.weight", {m_dim, m_ff_length}, TensorType::kQ8_0);
      layer.k_cache = allocate_f32("k_cache", {m_max_ctxt, m_num_kv_heads, m_key_length});
      layer.v_cache = allocate_f32("v_cache", {m_max_ctxt, m_num_kv_heads, m_value_length});
    }
  }

  u64 run(u64 input_token) {
    const u64 pos = m_context.size();
    extract_row(m_layer_xin, m_embedding, input_token);
    scale(m_layer_xin, m_layer_xin, std::sqrt(m_dim));

    for (auto& layer : m_layers) {
      copy(m_tcache.xskip, m_layer_xin);
      rms_norm(m_layer_xin, m_layer_xin, layer.attn_norm, m_rms_epsilon);

      // we compute the keys by multiplying with a weight matrix.
      // GEMMA3 uses multiple keys, one for each "kv head".
      // flatten to a single vector for multiplying:
      m_tcache.keys.reshape({m_num_kv_heads * m_key_length});
      matrix_vector_mt(m_tcache.keys, layer.wk, m_layer_xin, m_thread_pool);
      m_tcache.keys.reshape({m_num_kv_heads, m_key_length});

      m_tcache.values.reshape({m_num_kv_heads * m_value_length});
      matrix_vector_mt(m_tcache.values, layer.wv, m_layer_xin, m_thread_pool);
      m_tcache.values.reshape({m_num_kv_heads, m_value_length});

      m_tcache.queries.reshape({m_num_q_heads * m_key_length});
      matrix_vector_mt(m_tcache.queries, layer.wq, m_layer_xin, m_thread_pool);
      m_tcache.queries.reshape({m_num_q_heads, m_key_length});

      rms_norm_batch(m_tcache.keys, m_tcache.keys, layer.norm_k, m_rms_epsilon);
      rms_norm_batch(m_tcache.queries, m_tcache.queries, layer.norm_q, m_rms_epsilon);

      auto& rope_table = layer.local ? m_local_rope_table : m_global_rope_table;
      apply_rope_table_rows(m_tcache.keys, m_tcache.keys, rope_table, m_key_length, pos, false);
      apply_rope_table_rows(m_tcache.queries, m_tcache.queries, rope_table, m_key_length, pos,
                            false);

      // queries are scaled here:
      assert(m_key_length == 256);
      scale(m_tcache.queries, m_tcache.queries, 1.0f / std::sqrt(m_key_length));

      set_matrix(layer.k_cache, m_tcache.keys, pos);
      set_matrix(layer.v_cache, m_tcache.values, pos);

      u64 pos_start = 0;
      u64 pos_end = pos + 1;

      if (layer.local && (pos_end - pos_start) > m_sliding_window_size) {
        pos_start = pos_end - m_sliding_window_size;
      }

      // compute scores! Each row in the output is a score
      qk(m_tcache.scores, m_tcache.queries, layer.k_cache,
         {
             .context_length = m_max_ctxt,
             .num_q_heads = m_num_q_heads,
             .num_kv_heads = m_num_kv_heads,
             .key_length = m_key_length,
             .pos_0 = pos_start,
             .pos_1 = pos_end,
         });

      // do softmax over the scores
      softmax_cols_inplace(m_tcache.scores, pos_end - pos_start);

      m_tcache.context.reshape({m_num_q_heads, m_value_length});
      sv(m_tcache.context, m_tcache.scores, layer.v_cache,
         {
             .context_length = m_max_ctxt,
             .num_q_heads = m_num_q_heads,
             .num_kv_heads = m_num_kv_heads,
             .value_length = m_value_length,
             .pos_0 = pos_start,
             .pos_1 = pos_end,
         });

      // flatten back from multi-heads to a single vector
      m_tcache.context.reshape({m_num_q_heads * m_value_length});

      // final projection
      matrix_vector_mt(m_tcache.attn_result, layer.attn_output, m_tcache.context, m_thread_pool);
      rms_norm(m_tcache.attn_result, m_tcache.attn_result, layer.post_attn_norm, m_rms_epsilon);
      add(m_layer_xin, m_tcache.xskip, m_tcache.attn_result);

      // Feed forward

      copy(m_tcache.xskip, m_layer_xin);
      rms_norm(m_layer_xin, m_layer_xin, layer.ff_norm, m_rms_epsilon);
      matrix_vector_mt(m_tcache.up, layer.ff_up, m_layer_xin, m_thread_pool);
      matrix_vector_mt(m_tcache.gate, layer.ff_gate, m_layer_xin, m_thread_pool);
      gelu_tanh_approx(m_tcache.gate, m_ff_length);
      elementwise_multiply(m_tcache.up, m_tcache.gate);
      matrix_vector_mt(m_tcache.xff, layer.ff_down, m_tcache.up, m_thread_pool);
      rms_norm(m_tcache.xff, m_tcache.xff, layer.post_ff_norm, m_rms_epsilon);
      add(m_layer_xin, m_tcache.xskip, m_tcache.xff);
    }

    rms_norm(m_layer_xin, m_layer_xin, m_output_norm, m_rms_epsilon);
    matrix_vector_mt(m_logits, m_embedding, m_layer_xin, m_thread_pool);

    size_t max_index = sample_from_logits((float*)m_logits.data, m_vocab_size, kTemperature, m_rng);
    m_context.push_back(input_token);
    return max_index;
  }

  u64 run_batch(const u64* ptr, int n) {
    assert(n == kBatchSize);

    const u64 pos = m_context.size();

    // m_b.x_in = embedding[ids, :]
    extract_rows(m_b.x_in, m_embedding, ptr, n);
    scale(m_b.x_in, m_b.x_in, std::sqrt(m_dim));

    for (auto& layer : m_layers) {
      copy(m_b.skip, m_b.x_in);
      rms_norm_batch(m_b.x_in, m_b.x_in, layer.attn_norm, m_rms_epsilon);

      m_b.keys.reshape({kBatchSize, m_num_kv_heads * m_key_length});
      matrix_matrix_transpose(m_b.keys, layer.wk, m_b.x_in, m_thread_pool);
      m_b.keys.reshape({kBatchSize, m_num_kv_heads, m_key_length});

      m_b.values.reshape({kBatchSize, m_num_kv_heads * m_value_length});
      matrix_matrix_transpose(m_b.values, layer.wv, m_b.x_in, m_thread_pool);
      m_b.values.reshape({kBatchSize, m_num_kv_heads, m_value_length});

      m_b.queries.reshape({kBatchSize, m_num_q_heads * m_key_length});
      matrix_matrix_transpose(m_b.queries, layer.wq, m_b.x_in, m_thread_pool);
      m_b.queries.reshape({kBatchSize, m_num_q_heads, m_key_length});

      rms_norm_batch(m_b.keys, m_b.keys, layer.norm_k, m_rms_epsilon);
      rms_norm_batch(m_b.queries, m_b.queries, layer.norm_q, m_rms_epsilon);

      auto& rope_table = layer.local ? m_local_rope_table : m_global_rope_table;
      apply_rope_table_rows(m_b.keys, m_b.keys, rope_table, m_key_length, pos, true);
      apply_rope_table_rows(m_b.queries, m_b.queries, rope_table, m_key_length, pos, true);

      // queries are scaled here:
      assert(m_key_length == 256);
      scale(m_b.queries, m_b.queries, 1.0f / std::sqrt(m_key_length));

      set_matrices(layer.k_cache, m_b.keys, pos);
      set_matrices(layer.v_cache, m_b.values, pos);

      for (u64 i = 0; i < kBatchSize; i++) {
        u64 pos_start = 0;
        u64 pos_end = pos + 1 + i;

        // and for local layers, the position is limited to be only 1024 long.
        if (layer.local && (pos_end - pos_start) > m_sliding_window_size) {
          pos_start = pos_end - m_sliding_window_size;
        }
        // print("pos_start: {}, pos_end: {}\n", pos_start, pos_end);

        // compute scores! Each row in the output is a score
        qk_slice(m_tcache.scores, m_b.queries, layer.k_cache,
                 {
                     .context_length = m_max_ctxt,
                     .num_q_heads = m_num_q_heads,
                     .num_kv_heads = m_num_kv_heads,
                     .key_length = m_key_length,
                     .pos_0 = pos_start,
                     .pos_1 = pos_end,
                 },
                 i);

        // do softmax over the scores
        softmax_cols_inplace(m_tcache.scores, pos_end - pos_start);

        m_b.context.reshape({kBatchSize, m_num_q_heads, m_value_length});
        sv_slice(m_b.context, m_tcache.scores, layer.v_cache,
                 {
                     .context_length = m_max_ctxt,
                     .num_q_heads = m_num_q_heads,
                     .num_kv_heads = m_num_kv_heads,
                     .value_length = m_value_length,
                     .pos_0 = pos_start,
                     .pos_1 = pos_end,
                 },
                 i);
      }

      // flatten back from multi-heads to a single vector
      m_b.context.reshape({kBatchSize, m_num_q_heads * m_value_length});

      // final projection
      matrix_matrix_transpose(m_b.attn_result, layer.attn_output, m_b.context, m_thread_pool);
      // apply post-attention normalization
      rms_norm_batch(m_b.attn_result, m_b.attn_result, layer.post_attn_norm, m_rms_epsilon);

      // residual connection
      add(m_b.x_in, m_b.skip, m_b.attn_result);

      ///////////////
      // FEED FORWARD
      ///////////////

      // again, residual
      copy(m_b.skip, m_b.x_in);

      // apply normalization
      rms_norm_batch(m_b.x_in, m_b.x_in, layer.ff_norm, m_rms_epsilon);

      // feed forward
      matrix_matrix_transpose(m_b.up, layer.ff_up, m_b.x_in, m_thread_pool);
      matrix_matrix_transpose(m_b.gate, layer.ff_gate, m_b.x_in, m_thread_pool);
      gelu_tanh_approx(m_b.gate, m_ff_length * kBatchSize);
      elementwise_multiply(m_b.up, m_b.gate);
      matrix_matrix_transpose(m_b.xff, layer.ff_down, m_b.up, m_thread_pool);
      rms_norm_batch(m_b.xff, m_b.xff, layer.post_ff_norm, m_rms_epsilon);
      add(m_b.x_in, m_b.skip, m_b.xff);
    }

    // output norm
    rms_norm_batch(m_b.x_in, m_b.x_in, m_output_norm, m_rms_epsilon);
    extract_row(m_layer_xin, m_b.x_in, n - 1);
    matrix_vector_mt(m_logits, m_embedding, m_layer_xin, m_thread_pool);

    float* logits_data = (float*)m_logits.data;
    size_t max_index = sample_from_logits(logits_data, m_vocab_size, 1., m_rng);

    for (int i = 0; i < n; i++) {
      m_context.push_back(ptr[i]);
    }

    return max_index;
  }

  u64 total_context_length() const { return m_context.size(); }
  u64 max_context_length() const { return m_max_ctxt; }
  u64 model_context_length() const { return m_context_length; }

  Tensor load_tensor(const std::string& name,
                     const std::vector<u64>& shape,
                     gguf::TensorType type) {
    const auto& it = m_tensors.find(name);
    if (it == m_tensors.end()) {
      printf("tensor %s not found\n", name.c_str());
      assert(false);
    }
    const auto& tensor = it->second;
    assert(tensor.shape.size() == shape.size());
    // gguf has dims flipped, like (col, row)
    // but is still "row" major.

    for (u64 i = 0; i < tensor.shape.size(); i++) {
      assert(tensor.shape[i] == shape[tensor.shape.size() - i - 1]);
    }
    assert(tensor.type == type);
    Tensor result;
    switch (type) {
      case gguf::TensorType::kF32:
        result.type = Tensor::Type::F32;
        break;
      case gguf::TensorType::kQ8_0:
        result.type = Tensor::Type::GGUF_Q8_0;
        break;
      default:
        printf("unknown tensor type: %u\n", (u32)type);
        assert(false);
    }
    result.data = (void*)(m_tensor_data + tensor.offset);
    result.shape = shape;  // reversed! the right way!!
    return result;
  }

  Tensor allocate_f32(const std::string& name, const std::vector<u64>& shape) {
    return Tensor::allocate_f32(name, shape);
  }

 private:
  const u8* m_tensor_data;
  const std::unordered_map<std::string, gguf::Tensor>& m_tensors;

  u64 m_dim = 0;
  u64 m_num_q_heads = 0;
  u64 m_num_kv_heads = 0;
  u64 m_vocab_size = 0;
  u64 m_key_length = 0;
  u64 m_value_length = 0;
  u64 m_context_length = 0;
  u64 m_ff_length = 0;
  u64 m_sliding_window_size = 0;
  u64 m_max_ctxt = 0;
  std::vector<u64> m_context;
  float m_rms_epsilon = 0;
  Tensor m_layer_xin;          // dim
  Tensor m_logits;             // vocab_size
  Tensor m_local_rope_table;   // context_length, head_dim*2
  Tensor m_global_rope_table;  // head_dim * context_length
  Tensor m_embedding;          // vocab_size, dim
  Tensor m_output_norm;        // dim

  struct Layer {
    bool local = false;
    // norms
    Tensor attn_norm;       // dim
    Tensor post_attn_norm;  // dim
    Tensor norm_q;          // key_length
    Tensor norm_k;          // key_length
    Tensor ff_norm;         // dim
    Tensor post_ff_norm;    // dim
    // weight matrices
    Tensor wq;           // num_q_heads * key_length, dim
    Tensor wk;           // num_kv_heads * key_length, dim
    Tensor wv;           // num_kv_heads * value_length, dim
    Tensor attn_output;  // num_q_heads * value_length, dim
    Tensor ff_up;        // ff_length, dim
    Tensor ff_gate;      // ff_length, dim
    Tensor ff_down;      // dim, ff_length
    // kv
    Tensor k_cache;  // context_length, num_kv_heads, key_length
    Tensor v_cache;  // context_length, num_kv_heads, value_length
  };
  std::vector<Layer> m_layers;

  struct TCache {
    Tensor xskip;    // dim
    Tensor xattn;    // dim
    Tensor xff;      // dim
    Tensor queries;  // num_q_heads * key_length
    Tensor keys;     // num_kv_heads * key_length
    Tensor values;   // num_kv_heads * value_length
    // not sure if scores should be this way or not.
    Tensor scores;       // context_length * num_q_heads
    Tensor context;      // num_q_heads * value_length
    Tensor attn_result;  // dim
    Tensor up;           // ff_length
    Tensor gate;         // ff_length
  } m_tcache;

  struct BatchCache {
    Tensor x_in;         // batch_size, dim
    Tensor skip;         // batch_size, dim
    Tensor keys;         // batch_size, num_kv_heads, key_length
    Tensor values;       // batch_size, num_kv_heads, value_length
    Tensor queries;      // batch_size, num_q_heads, key_length
    Tensor context;      // batch_size, num_q_heads, value_length
    Tensor attn_result;  // batch_size, dim
    Tensor up;           // batch_size, ff_length
    Tensor gate;         // batch_size, ff_length
    Tensor xff;          // batch_size, dim
  } m_b;

  ThreadPool m_thread_pool;
  std::default_random_engine m_rng;
};
