#pragma once

#include "llm/model.h"
#include "llm/tokenizer.h"

void chatbot(Tokenizer& tokenizer, Model& model);
std::pair<const void*, size_t> map_file(const char* path);
int run(const char* filename);
