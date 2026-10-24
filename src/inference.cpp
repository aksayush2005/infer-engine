#include "llm/inference.h"

void chatbot(Tokenizer& tokenizer, Model& model) {
  std::vector<u64> context;

  {
    std::string system_input(kSystemPrompt);
    if (!system_input.empty()) {
      for (auto tok : tokenizer.tokenize(system_input)) {
        model.run(tok);
      }
    }
  }

  while (true) {
    u64 newline = tokenizer.find_token("\n").value();
    std::string user_input;
    std::cout << "> ";
    std::getline(std::cin, user_input);
    if (user_input == "quit")
      break;

    if (user_input.empty())
      continue;

    std::vector<u64> tokens;
    tokens.push_back(tokenizer.start_of_turn_token());
    tokens.push_back(tokenizer.user_token());
    tokens.push_back(newline);
    auto user_tokens = tokenizer.tokenize(user_input);
    tokens.insert(tokens.end(), user_tokens.begin(), user_tokens.end());
    tokens.push_back(tokenizer.end_of_turn_token());
    tokens.push_back(newline);
    tokens.push_back(tokenizer.start_of_turn_token());
    tokens.push_back(tokenizer.model_token());
    tokens.push_back(newline);

    u64 next = -1;
    u64 i = 0;
    auto start = std::chrono::high_resolution_clock::now();
    u64 batch_iters = 0;
    while (i + kBatchSize <= tokens.size()) {
      next = model.run_batch(tokens.data() + i, kBatchSize);
      i += kBatchSize;
      batch_iters++;
    }
    while (i < tokens.size()) {
      next = model.run(tokens[i]);
      i++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (kShowStats) {
      std::cout << "batch time for " << batch_iters << " batches: " << duration.count() << " ms / "
                << tokens.size() << " tokens = " << (int)(duration.count() / (float)tokens.size())
                << " ms/token\n";
      std::cout << "---------------------------------------------\n";
    }

    start = std::chrono::high_resolution_clock::now();
    int ntok = 0;
    while (next != tokenizer.end_of_turn_token()) {
      std::cout << tokenizer.token(next) << std::flush;
      next = model.run(next);
      ntok++;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\n";

    if (kShowStats) {
      std::cout << "time: " << duration.count() << " ms / " << ntok
                << " tokens = " << (int)(duration.count() / (float)ntok) << " ms/token\n";
      std::cout << "used " << model.total_context_length() << "/" << model.max_context_length()
                << " tokens (model supports " << model.model_context_length() << ")\n";
      std::cout << "---------------------------------------------\n";
    }
  }
}

std::pair<const void*, size_t> map_file(const char* path) {
  int fd = open(path, O_RDONLY);
  assert(fd != -1);
  struct stat st;
  int ret = fstat(fd, &st);
  assert(ret != -1);
  size_t size = st.st_size;
  void* addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  assert(addr != MAP_FAILED);
  close(fd);
  return {addr, size};
}

int run(const char* filename) {
  printf("Opening model file...\n");
  auto [data, size] = map_file(filename);
  printf("Loading metadata...\n");
  FileStream stream(data);
  auto header = stream.read<gguf::Header>();
  auto metadata = gguf::read_metadata(&stream, header.metadata_kv_count);
  printf(" read %ld metadata values\n", header.metadata_kv_count);
  auto tensors = gguf::read_tensors(&stream, header.tensor_count);
  printf(" read %ld tensors\n", header.tensor_count);

  printf("Creating tokenizer...\n");
  Tokenizer tokenizer(metadata.at("tokenizer.ggml.tokens").array,
                      metadata.at("tokenizer.ggml.scores").array);

  const u64 offset = stream.pos;
  const u64 aligned_offset = (offset + 31) & ~31;
  printf("Loading model...\n");
  Model model((const u8*)stream.data + aligned_offset, tensors, metadata);
  model.run(tokenizer.bos_token());

  chatbot(tokenizer, model);
  return 0;
}

