#pragma once

#include "llm/common.h"

class ThreadPool {
 public:
  explicit ThreadPool(size_t thread_count) {
    for (size_t i = 0; i < thread_count; ++i) {
      m_workers.emplace_back([this] {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cv.wait(lock, [this] { return m_stop || !m_tasks.empty(); });

            if (m_stop && m_tasks.empty())
              return;

            task = std::move(m_tasks.front());
            m_tasks.pop();
          }
          task();

          std::lock_guard<std::mutex> lock(m_mtx);
          m_pending_tasks--;
          if (m_pending_tasks == 0)
            m_wait_cv.notify_all();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(m_mtx);
      m_stop = true;
    }
    m_cv.notify_all();
    for (auto& t : m_workers)
      t.join();
  }

  void enqueue(std::function<void()> fn) {
    {
      std::unique_lock<std::mutex> lock(m_mtx);
      m_pending_tasks++;
      m_tasks.push(std::move(fn));
    }
    m_cv.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(m_mtx);
    m_wait_cv.wait(lock, [this] { return m_pending_tasks == 0; });
  }

  u64 num_threads() const { return m_workers.size(); }

 private:
  std::vector<std::thread> m_workers;
  std::queue<std::function<void()>> m_tasks;
  std::mutex m_mtx;
  std::condition_variable m_cv;
  bool m_stop = false;
  int m_pending_tasks = 0;
  // std::mutex m_wait_mtx;
  std::condition_variable m_wait_cv;
};
