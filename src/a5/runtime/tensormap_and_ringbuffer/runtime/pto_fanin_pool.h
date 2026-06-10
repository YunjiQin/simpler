/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * PTO2FaninPool — per-ring spill ring buffer for fanin entries.
 *
 * Lives in its own header so PTO2SharedMemoryRingHeader can embed it without
 * pulling in pto_ring_buffer.h (which itself references the SM ring header).
 *
 * Ownership: per-ring shared resource. Orchestrator writes (alloc + advance_tail),
 * wiring thread and scheduler read via the embedded copy on the SM ring header.
 */

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_FANIN_POOL_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_FANIN_POOL_H_

#include <algorithm>
#include <atomic>
#include <inttypes.h>
#include <stdint.h>
#include <type_traits>

#include "common/unified_log.h"
#include "pto_runtime_status.h"
#include "pto_runtime2_types.h"  // PTO2FaninSpillEntry, PTO2TaskPayload, PTO2_FANIN_INLINE_CAP

// Forward decl — reclaim/ensure_space take a SM ring header reference, but the
// bodies live in pto_ring_buffer.cpp where the full type is available.
struct PTO2SharedMemoryRingHeader;

/**
 * Fanin spill pool structure
 *
 * True ring buffer for allocating spilled fanin entries.
 * Entries are reclaimed when their consumer tasks become CONSUMED.
 *
 * Linear counters (top, tail) grow monotonically; the physical index
 * is obtained via modulo: base[linear_index % capacity].
 */
struct PTO2FaninPool {
    PTO2FaninSpillEntry *base;       // Pool base address
    int32_t capacity;                // Total number of entries
    int32_t top;                     // Linear next-allocation counter (starts from 1)
    int32_t tail;                    // Linear first-alive counter (entries before this are dead)
    int32_t high_water;              // Peak concurrent usage (top - tail)
    int32_t reclaim_task_cursor{0};  // Last task id scanned for reclaim on this pool

    std::atomic<int32_t> *error_code_ptr = nullptr;

    void init(PTO2FaninSpillEntry *in_base, int32_t in_capacity, std::atomic<int32_t> *in_error_code_ptr) {
        base = in_base;
        capacity = in_capacity;
        top = 1;
        tail = 1;
        high_water = 0;
        reclaim_task_cursor = 0;
        base[0].slot_state = nullptr;
        error_code_ptr = in_error_code_ptr;
    }

    void reclaim(PTO2SharedMemoryRingHeader &ring, int32_t sm_last_task_alive);

    bool ensure_space(PTO2SharedMemoryRingHeader &ring, int32_t needed);

    PTO2FaninSpillEntry *alloc() {
        int32_t used = top - tail;
        if (used >= capacity) {
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Fanin Spill Pool Overflow!");
            LOG_ERROR("========================================");
            LOG_ERROR("Fanin spill pool exhausted: %d entries alive (capacity=%d).", used, capacity);
            LOG_ERROR("  - Pool top:      %d (linear)", top);
            LOG_ERROR("  - Pool tail:     %d (linear)", tail);
            LOG_ERROR("  - High water:    %d", high_water);
            LOG_ERROR("Solution:");
            LOG_ERROR("  Increase fanin spill pool capacity (current: %d, recommended: %d).", capacity, capacity * 2);
            LOG_ERROR("  Compile-time: PTO2_DEP_LIST_POOL_SIZE in pto_runtime2_types.h");
            LOG_ERROR("  Runtime env:  PTO2_RING_DEP_POOL=%d", capacity * 2);
            LOG_ERROR("========================================");
            if (error_code_ptr) {
                error_code_ptr->store(PTO2_ERROR_DEP_POOL_OVERFLOW, std::memory_order_release);
            }
            return nullptr;
        }
        int32_t idx = top % capacity;
        top++;
        used++;
        if (used > high_water) high_water = used;
        return &base[idx];
    }

    void advance_tail(int32_t new_tail) {
        if (new_tail > tail) {
            tail = new_tail;
        }
    }

    int32_t used() const { return top - tail; }

    int32_t available() const { return capacity - used(); }
};

// =============================================================================
// Fanin Iteration Helpers
// =============================================================================
//
// Walk the fanin list for a task: inline prefix first (payload.fanin_inline_slot_states[]),
// then any spill in the per-ring PTO2FaninPool. Callback may return void (always
// continue) or bool (false = stop, propagated as the helper's return).

template <typename Fn>
using PTO2FaninCallbackResult = std::invoke_result_t<Fn &, PTO2TaskSlotState *>;

template <typename Fn>
using PTO2FaninForEachReturn = std::conditional_t<std::is_same_v<PTO2FaninCallbackResult<Fn>, void>, void, bool>;

template <typename InlineSlots, typename Fn>
inline PTO2FaninForEachReturn<Fn> for_each_fanin_storage(
    InlineSlots &&inline_slot_states, int32_t fanin_count, int32_t spill_start, PTO2FaninPool &spill_pool, Fn &&fn
) {
    using FaninCallbackResult = PTO2FaninCallbackResult<Fn>;
    static_assert(
        std::is_same_v<FaninCallbackResult, void> || std::is_same_v<FaninCallbackResult, bool>,
        "fanin callback must return void or bool"
    );

    if constexpr (std::is_void_v<FaninCallbackResult>) {
        int32_t inline_count = std::min(fanin_count, PTO2_FANIN_INLINE_CAP);
        for (int32_t i = 0; i < inline_count; i++) {
            fn(inline_slot_states[i]);
        }

        int32_t spill_count = fanin_count - inline_count;
        if (spill_count <= 0) {
            return;
        }

        int32_t start_idx = spill_start % spill_pool.capacity;
        int32_t first_count = std::min(spill_count, spill_pool.capacity - start_idx);
        PTO2FaninSpillEntry *first = spill_pool.base + start_idx;
        for (int32_t i = 0; i < first_count; i++) {
            fn(first[i].slot_state);
        }

        int32_t second_count = spill_count - first_count;
        for (int32_t i = 0; i < second_count; i++) {
            fn(spill_pool.base[i].slot_state);
        }
        return;
    } else {
        int32_t inline_count = std::min(fanin_count, PTO2_FANIN_INLINE_CAP);
        for (int32_t i = 0; i < inline_count; i++) {
            if (!fn(inline_slot_states[i])) {
                return false;
            }
        }

        int32_t spill_count = fanin_count - inline_count;
        if (spill_count <= 0) {
            return true;
        }

        int32_t start_idx = spill_start % spill_pool.capacity;
        int32_t first_count = std::min(spill_count, spill_pool.capacity - start_idx);
        PTO2FaninSpillEntry *first = spill_pool.base + start_idx;
        for (int32_t i = 0; i < first_count; i++) {
            if (!fn(first[i].slot_state)) {
                return false;
            }
        }

        int32_t second_count = spill_count - first_count;
        for (int32_t i = 0; i < second_count; i++) {
            if (!fn(spill_pool.base[i].slot_state)) {
                return false;
            }
        }
        return true;
    }
}

template <typename Fn>
inline PTO2FaninForEachReturn<Fn> for_each_fanin_slot_state(
    const PTO2TaskPayload &payload, PTO2FaninPool &spill_pool, Fn &&fn
) {
    return for_each_fanin_storage(
        payload.fanin_inline_slot_states, payload.fanin_actual_count, payload.fanin_spill_start, spill_pool,
        static_cast<Fn &&>(fn)
    );
}

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_FANIN_POOL_H_
