#include "llm/inference.h"

int main(int argc, char** argv) {
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  if (argc != 2 && argc != 3) {
    printf("Usage: llm model_file.gguf\n");
    return 1;
  }
  return run(argv[1]);
}