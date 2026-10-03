# LLM Inference Stack from Scratch

A from-scratch implementation of a complete LLM inference and serving stack, built to understand the core systems behind modern inference engines such as vLLM.

The project implements the major components required to turn a causal Transformer into a multi-request inference service:

- Tokenization
- Autoregressive sampling
- Causal Transformer inference
- KV caching
- Paged attention
- KV-cache memory allocation
- Continuous batching
- Request scheduling
- Streaming generation
- HTTP serving API
- Throughput and latency benchmarking

The goal is to study LLM inference as a **systems problem**, where compute, GPU memory, scheduling, batching, and request-level latency interact.

---

## Architecture

```text
                         Client Requests
                                |
                                v
                    +-----------------------+
                    |    Serving API        |
                    |   Streaming / HTTP    |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | Request Manager       |
                    | Queue + Lifecycle     |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | Scheduler             |
                    | Continuous Batching   |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | Batch Manager         |
                    | Active Sequences      |
                    +-----------+-----------+
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
             Tokenization              Sampling
                    |                       |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | Transformer           |
                    | Inference             |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | KV Cache Manager      |
                    | Paged Allocation      |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    | Paged Attention       |
                    +-----------+-----------+
                                |
                                v
                         Generated Tokens
                                |
                                v
                       Streaming Response
```

---

## Core Components

### Tokenization

The inference stack includes tokenization and detokenization so that the serving layer can operate directly on user text.

The tokenizer is responsible for converting:

```text
Text
  |
  v
Token IDs
```

and the reverse operation:

```text
Token IDs
  |
  v
Generated Text
```

The implementation keeps tokenization separate from model execution so that the inference engine operates on integer token sequences.

---

## Transformer Inference

A small causal Transformer is implemented to provide a controlled environment for studying inference behavior.

The model contains the standard components:

- Token embeddings
- Positional representation
- Multi-head self-attention
- Feed-forward layers
- Residual connections
- Layer normalization
- Language-model output head

Inference is autoregressive:

```text
Prompt
  |
  v
Transformer
  |
  v
Next-token distribution
  |
  v
Sampler
  |
  v
Next token
  |
  +----> Transformer
```

The model is intentionally small enough to make the complete inference stack understandable and measurable.

---

## KV Cache

Recomputing the entire sequence during every decoding step is unnecessarily expensive.

Instead, the inference engine stores previously computed keys and values:

```text
Sequence:
[t1, t2, t3, t4]

KV Cache:
K1 K2 K3 K4
V1 V2 V3 V4
```

When a new token arrives, only the new key and value need to be computed.

This changes decoding from repeatedly processing the entire prefix to incrementally extending the cached state.

The KV cache becomes one of the primary memory-management problems in the serving system.

---

## Paged KV Cache

A conventional KV cache can allocate a contiguous region for every sequence.

This can cause memory fragmentation and makes dynamic request management difficult.

The implementation instead divides KV-cache memory into fixed-size blocks:

```text
GPU KV Memory

+--------+--------+--------+--------+--------+
| Block0 | Block1 | Block2 | Block3 | Block4 |
+--------+--------+--------+--------+--------+
```

Each sequence maintains a logical mapping from its token positions to physical KV blocks:

```text
Sequence A:

Token 0-15   -> Block 3
Token 16-31  -> Block 7
Token 32-47  -> Block 1


Sequence B:

Token 0-15   -> Block 5
Token 16-31  -> Block 2
```

This allows KV memory to be allocated and released dynamically without requiring each sequence to occupy one contiguous region.

---

## Paged Attention

Paged attention performs attention over KV blocks rather than assuming that a sequence's KV cache is physically contiguous.

Conceptually:

```text
Query
 |
 v
+----------------------+
| Block Table           |
+----------+-----------+
           |
     +-----+-----+-----+
     |           |     |
     v           v     v
   Block 3     Block 7 Block 1
     |           |     |
     +-----+-----+-----+
           |
           v
      Attention
           |
           v
       Output
```

The attention computation uses the logical sequence-to-block mapping maintained by the KV-cache allocator.

This separates:

```text
Logical sequence layout
```

from:

```text
Physical GPU memory layout
```

which is fundamental to efficient serving under dynamic workloads.

---

## KV-Cache Allocator

The allocator maintains the available KV-cache blocks and assigns them to active requests.

A simplified lifecycle is:

```text
Request arrives
      |
      v
Allocate blocks
      |
      v
Generate tokens
      |
      v
Allocate additional blocks
      |
      v
Request completes
      |
      v
Release blocks
      |
      v
Return blocks to free pool
```

The allocator must handle:

- Block allocation
- Block release
- Sequence growth
- Block-table management
- Free-block tracking
- Memory exhaustion
- Variable sequence lengths

This provides a small-scale implementation of the memory-management problems encountered by production LLM serving engines.

---

## Continuous Batching

Traditional static batching waits for a group of requests to form a batch:

```text
Request A ----+
Request B ----+---- Batch ----> Model
Request C ----+
```

This becomes inefficient when requests have different generation lengths.

Continuous batching instead allows requests to enter and leave the active batch dynamically:

```text
Time ------------------------------------------------>

Request A: [====================]
Request B:      [==========]
Request C:          [================]
Request D:                 [=======]
```

At each decoding iteration, the scheduler constructs the current active batch.

Completed sequences can leave while new requests enter.

This improves GPU utilization under dynamic workloads.

---

## Request Scheduling

The scheduler controls which requests participate in each inference iteration.

A scheduling iteration roughly performs:

```text
1. Inspect waiting requests
2. Inspect active requests
3. Check available KV-cache blocks
4. Admit new requests when resources permit
5. Select active sequences
6. Construct the batch
7. Execute model inference
8. Sample next tokens
9. Update KV-cache state
10. Stream completed tokens
11. Remove finished requests
```

The scheduler therefore connects request-level behavior with GPU and KV-cache resource constraints.

---

## Sampling

The inference engine supports autoregressive token sampling from the model's output distribution.

The sampling pipeline can include:

```text
Logits
  |
  v
Temperature
  |
  v
Top-K
  |
  v
Top-P
  |
  v
Probability Distribution
  |
  v
Sample Token
```

Greedy decoding is also supported for deterministic inference.

The separation between model logits and sampling makes it possible to benchmark different decoding strategies independently of the Transformer.

---

## Streaming Generation

Generated tokens are exposed incrementally rather than waiting for the entire sequence to finish.

Conceptually:

```text
Request
  |
  v
Generate token 1 -> Stream
  |
  v
Generate token 2 -> Stream
  |
  v
Generate token 3 -> Stream
  |
  v
...
```

The serving layer can therefore provide time-to-first-token and inter-token latency measurements.

---

## Serving API

The inference engine exposes a streaming API for submitting generation requests.

A request contains information such as:

```json
{
  "prompt": "Explain how KV caching works",
  "max_tokens": 128,
  "temperature": 0.8
}
```

The server manages the request lifecycle:

```text
HTTP Request
     |
     v
Tokenization
     |
     v
Request Queue
     |
     v
Scheduler
     |
     v
Inference Engine
     |
     v
Streaming Tokens
     |
     v
HTTP Response
```

The API layer is intentionally separated from the inference engine so that scheduling and model execution can be tested independently.

---

## Inference Loop

The central serving loop continuously processes active requests:

```text
while server_is_running:

    receive_new_requests()

    tokenize_requests()

    schedule_requests()

    allocate_kv_blocks()

    run_model_forward()

    sample_tokens()

    update_kv_cache()

    stream_tokens()

    release_finished_requests()
```

Unlike offline generation, the engine must continuously make decisions while requests arrive, generate tokens, and finish at different times.

---

## Performance Benchmarking

The benchmark harness evaluates both **latency** and **throughput** under different request workloads.

Important metrics include:

| Metric | Description |
|---|---|
| TTFT | Time to first generated token |
| ITL | Inter-token latency |
| End-to-End Latency | Total request latency |
| Throughput | Generated tokens per second |
| Request Throughput | Requests completed per second |
| Batch Size | Number of active sequences |
| KV Memory Usage | KV-cache consumption |
| Scheduler Time | Time spent scheduling |
| Model Time | Transformer execution time |
| Allocation Time | KV-cache allocation overhead |

The benchmark can vary:

- Number of concurrent requests
- Prompt length
- Generation length
- Batch size
- KV-cache block size
- Sampling configuration
- Request arrival patterns

---

## Latency and Throughput

The benchmark distinguishes between two important serving objectives.

### Time to First Token

TTFT measures the latency between receiving a request and producing its first generated token.

```text
Request
  |
  +----------------------+
                         |
                         v
                    First Token
```

### Inter-Token Latency

ITL measures the time between successive generated tokens.

```text
Token 1 ---- Token 2 ---- Token 3 ---- Token 4
      |             |             |
     ITL           ITL           ITL
```

### Throughput

Throughput measures how much useful generation the system can sustain:

```text
Throughput = Generated Tokens / Total Time
```

Optimizing a serving system therefore requires balancing latency and aggregate throughput rather than optimizing a single metric.

---

## Memory Model

The project treats KV-cache memory as a managed resource.

```text
                GPU Memory
                    |
        +-----------+-----------+
        |                       |
        v                       v
   Model Weights            KV Cache
                                |
                       +--------+--------+
                       |        |        |
                    Block 0   Block 1  Block 2
                       |        |        |
                       +--------+--------+
                                |
                         Block Tables
                                |
                         Active Requests
```

Paged allocation allows multiple sequences to share the global KV-cache memory pool without requiring contiguous per-sequence allocations.

---

## System Design

The complete inference stack can be viewed as several layers:

```text
+--------------------------------------------------+
|                  Streaming API                   |
+--------------------------------------------------+
|                Request Manager                   |
+--------------------------------------------------+
|                   Scheduler                      |
+--------------------------------------------------+
|                Continuous Batch                  |
+--------------------------------------------------+
|             Inference Runtime                    |
+--------------------------------------------------+
|           KV Cache / Block Allocator             |
+--------------------------------------------------+
|                Paged Attention                   |
+--------------------------------------------------+
|              Transformer Model                   |
+--------------------------------------------------+
|          GPU / PyTorch / Kernels                 |
+--------------------------------------------------+
```

Each layer has a distinct responsibility while communicating with the layers below it.

---

## Engineering Focus

The project focuses on the systems problems that appear when moving from a single-model inference script to a multi-request LLM serving engine:

- GPU memory management
- KV-cache allocation
- Memory fragmentation
- Dynamic batching
- Request scheduling
- Variable sequence lengths
- Streaming responses
- Sampling
- Queueing behavior
- Latency vs throughput trade-offs
- GPU utilization
- End-to-end inference performance

The implementation is intentionally small enough to inspect end-to-end while retaining the architectural ideas used by modern LLM inference systems.

---

## Technology Stack

- Python
- PyTorch
- CUDA
- Triton
- FastAPI
- NumPy

---

## Concepts Covered

```text
LLM Inference
    |
    +-- Tokenization
    +-- Sampling
    +-- Transformer Inference
    +-- KV Caching
    |
    v
Memory Management
    |
    +-- KV Cache Blocks
    +-- Block Tables
    +-- Memory Allocation
    +-- Memory Reuse
    |
    v
Serving Systems
    |
    +-- Request Queues
    +-- Scheduling
    +-- Continuous Batching
    +-- Streaming
    |
    v
Performance Engineering
    |
    +-- TTFT
    +-- ITL
    +-- Throughput
    +-- GPU Utilization
    +-- Latency
```

---

## References

- Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention."
- Kwon, W. et al. "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention."
- Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."
- Vaswani, A. et al. "Attention Is All You Need."

---

## License

See the repository license for applicable terms.
