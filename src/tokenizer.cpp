#include "llm/tokenizer.h"

std::vector<std::string> utf8_split(const std::string& text) {
  std::vector<std::string> characters;
  u64 pos = 0;
  while (pos < text.length()) {
    const char* ptr = text.c_str() + pos;
    u8 c = *ptr;
    if ((c & 0x80) == 0) {
      characters.push_back(std::string(ptr, ptr + 1));
      pos++;
    } else if ((c & 0xE0) == 0xC0) {
      assert(pos + 1 < text.length());
      characters.push_back(std::string(ptr, ptr + 2));
      pos += 2;
    } else if ((c & 0xF0) == 0xE0) {
      assert(pos + 2 < text.length());
      characters.push_back(std::string(ptr, ptr + 3));
      pos += 3;
    } else if ((c & 0xF8) == 0xF0) {
      assert(pos + 3 < text.length());
      characters.push_back(std::string(ptr, ptr + 4));
      pos += 4;
    } else {
      assert(false);
    }
  }
  return characters;
}
