#pragma once

#include <cstdint>
#include <vector>

namespace minidecode {

class BlockManager {
   public:
    explicit BlockManager(int num_blocks);
    int allocate();
    void free(int block_id);

    int num_free_blocks() const noexcept { return m_free_blocks.size(); }
    int num_total_blocks() const noexcept { return m_total_blocks; }
    bool is_allocated(int block_id) const;

   private:
    int m_total_blocks;
    std::vector<int> m_free_blocks;
    std::vector<std::uint8_t> m_allocated;
};

}  // namespace minidecode