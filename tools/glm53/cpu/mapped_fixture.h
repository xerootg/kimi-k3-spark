// Test/benchmark weights use the same read-only file mappings as the server.
#pragma once
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
struct MappedFixture {
  const uint8_t *ptr;
  size_t bytes;
  explicit MappedFixture(std::vector<uint8_t> &source) : bytes(source.size()) {
    char name[] = "tools/glm53/cpu/build/tmp/fixture-XXXXXX";
    int fd = mkstemp(name);
    if (fd < 0)
      throw std::runtime_error("fixture open");
    unlink(name);
    size_t off = 0;
    while (off < bytes) {
      ssize_t n = write(fd, source.data() + off, bytes - off);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0) {
        close(fd);
        throw std::runtime_error("fixture write");
      }
      off += n;
    }
    void *map =
        mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED)
      throw std::runtime_error("fixture mmap");
    ptr = static_cast<const uint8_t *>(map);
    std::vector<uint8_t>().swap(source);
  }
  const uint8_t *data() const { return ptr; }
  ~MappedFixture() { munmap(const_cast<uint8_t *>(ptr), bytes); }
  MappedFixture(const MappedFixture &) = delete;
  MappedFixture &operator=(const MappedFixture &) = delete;
};
