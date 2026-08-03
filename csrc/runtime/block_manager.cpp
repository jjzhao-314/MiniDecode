#include "block_manager.h"

#include <stdexcept>

minidecode::BlockManager::BlockManager(int num_blocks)
    : m_total_blocks(num_blocks) {
    if (num_blocks <= 0) {
        throw std::invalid_argument("num_blocks must be positive");
    }
    m_free_blocks.reserve(num_blocks);
    m_allocated.resize(num_blocks, 0);
    for (int i = 0; i < num_blocks; ++i) {
        m_free_blocks.push_back(i);
    }
}

int minidecode::BlockManager::allocate() {
    if (m_free_blocks.empty()) throw std::runtime_error("no block left.");
    int block_id = m_free_blocks.back();
    if (m_allocated[block_id] == 1) throw std::runtime_error("internal error.");
    m_allocated[block_id] = 1;
    m_free_blocks.pop_back();
    return block_id;
}

void minidecode::BlockManager::free(int block_id) {
    if (block_id < 0 || block_id >= m_total_blocks)
        throw std::out_of_range("invalid block id");
    if (m_allocated[block_id] == 0) throw std::runtime_error("internal error.");
    m_allocated[block_id] = 0;
    m_free_blocks.push_back(block_id);
}

bool minidecode::BlockManager::is_allocated(int block_id) const {
    if (block_id < 0 || block_id >= m_total_blocks) {
        throw std::out_of_range("invalid block id");
    }
    return m_allocated[block_id] == 1;
}
