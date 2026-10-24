#pragma once

#include "llm/gguf.h"

std::vector<std::string> utf8_split(const std::string& text);

class Tokenizer {
 public:
  Tokenizer(const std::vector<gguf::Value>& tokens, const std::vector<gguf::Value>& scores) {
    auto add_token = [&](const std::string& token) {
      const u64 id = m_id_to_string.size();
      const auto& [it, inserted] = m_string_to_id.insert({token, id});
      assert(inserted);
      m_id_to_string.push_back(it->first);
    };

    for (u64 i = 0; i < tokens.size(); i++) {
      if (tokens[i].string.find("â–") == 0) {
        std::string fixed;
        fixed += ' ';
        for (size_t j = strlen("â–"); j < tokens[i].string.length(); j++) {
          fixed += tokens[i].string[j];
        }
        add_token(fixed);
      } else {
        add_token(tokens[i].string);
      }
    }
    assert(m_string_to_id.size() == tokens.size());

    m_scores.reserve(scores.size());
    for (const auto& score : scores) {
      m_scores.push_back(score.f32);
    }

    m_start_of_turn_token = find_token("<start_of_turn>").value();
    m_end_of_turn_token = find_token("<end_of_turn>").value();
    m_user_token = find_token("user").value();
    m_model_token = find_token("model").value();
    m_bos_token = find_token("<bos>").value();
  }

  std::vector<u64> tokenize(const std::string& text) const {
    std::vector<u64> tokens;
    std::vector<std::string> characters = utf8_split(text);

    int consecutive_spaces = 0;
    auto flush_spaces = [&]() {
      while (consecutive_spaces > 0) {
        int to_flush = std::min(consecutive_spaces, 31);
        tokens.push_back(find_token(std::string(to_flush, ' ')).value());
        consecutive_spaces -= to_flush;
      }
    };
    for (const auto& c : characters) {
      if (c == " ") {
        consecutive_spaces++;
      } else {
        if (consecutive_spaces > 0) {
          flush_spaces();
        }
        tokens.push_back(find_token(c).value());
      }
    }
    if (consecutive_spaces > 0) {
      flush_spaces();
    }

    while (true) {
      s64 best_idx = -1;
      float best_score = 0;
      u64 best_merged_token = 0;

      // try all pairs of tokens, and find the best to merge
      for (u64 i = 0; i < tokens.size() - 1; i++) {
        // look up the merged token
        if (auto merged_token =
                find_token(m_id_to_string[tokens[i]] + m_id_to_string[tokens[i + 1]])) {
          // update if it's better
          if (best_idx < 0 || m_scores[merged_token.value()] > best_score) {
            best_score = m_scores[merged_token.value()];
            best_idx = i;
            best_merged_token = merged_token.value();
          }
        }
      }

      if (best_idx < 0) {
        // no merges possible, we're done
        break;
      }

      // merge the best pair
      tokens[best_idx] = best_merged_token;
      tokens.erase(tokens.begin() + best_idx + 1);
    }
    return tokens;
  }
  std::optional<u64> find_token(const std::string& text) const {
    auto it = m_string_to_id.find(text);
    if (it == m_string_to_id.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  const std::string& token(u64 idx) const { return m_id_to_string[idx]; }
  std::string decode(const std::vector<u64>& tokens) const {
    std::string ret;
    for (auto token : tokens) {
      ret += m_id_to_string[token];
    }
    return ret;
  }
  u64 start_of_turn_token() const { return m_start_of_turn_token; }
  u64 end_of_turn_token() const { return m_end_of_turn_token; }
  u64 user_token() const { return m_user_token; }
  u64 model_token() const { return m_model_token; }
  u64 bos_token() const { return m_bos_token; }

 private:
  std::unordered_map<std::string, u64> m_string_to_id;
  std::vector<std::string> m_id_to_string;
  std::vector<float> m_scores;
  u64 m_start_of_turn_token = 0;
  u64 m_end_of_turn_token = 0;
  u64 m_user_token = 0;
  u64 m_model_token = 0;
  u64 m_bos_token = 0;
};

