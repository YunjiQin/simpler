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
#pragma once

#include <algorithm>
#include <vector>

#include "chip_callable_layout.h"
#include "callable_protocol.h"
#include "runtime_c_api.h"

struct KernelCallableResidency {
    int32_t callable_id{-1};
    uint64_t device_address{0};
    size_t bytes{0};
};

// The caller serializes prepare/resolve/close. Entries and arena addresses
// remain immutable until external quiescence permits context close.
class KernelCallableCache {
public:
    static constexpr size_t kByteLimit = 512ULL * 1024 * 1024;

    explicit KernelCallableCache(size_t byte_limit = kByteLimit) :
        byte_limit_(std::min(byte_limit, kByteLimit)) {}

    struct Ops {
        void *context;
        void *(*allocate)(void *, size_t);
        int (*copy)(void *, void *, const void *, size_t);
    };

    /**
     * Admit one callable image and mint the id it becomes resident under.
     *
     * Registration is not deduplicated: the same image admitted twice takes
     * two ids, two uploads and two charges against the arena, so capacity is
     * spent per registration rather than per distinct image. An id is minted
     * once and never reused for the life of the context, which is what makes
     * a resolve unambiguous without a generation comparand.
     *
     * A staged entry must be committed or rolled back before the next
     * admission; only the most recent entry can be rolled back.
     */
    int stage(const ChipCallable *callable, size_t bytes, const Ops &ops, int32_t &out_callable_id) {
        out_callable_id = -1;
        if (bytes > byte_limit_) return PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED;
        int rc = validate_image(callable, bytes);
        if (rc != 0) return rc;
        const auto layout = compute_chip_callable_layout(callable);
        if (!entries_.empty() && !entries_.back().ready) return PTO_RUNTIME_ERR_INVALID_STATE;
        if (entries_.size() >= MAX_REGISTERED_CALLABLE_IDS) return PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED;
        const size_t padding = (CALLABLE_ALIGN - bytes % CALLABLE_ALIGN) % CALLABLE_ALIGN;
        if (bytes > byte_limit_ - used_ || padding > byte_limit_ - used_ - bytes)
            return PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED;
        const size_t charged = bytes + padding;
        const auto id = static_cast<int32_t>(entries_.size());
        Entry candidate;
        candidate.residency = {id, 0, bytes};
        candidate.image = std::vector<uint8_t>(
            reinterpret_cast<const uint8_t *>(callable), reinterpret_cast<const uint8_t *>(callable) + bytes
        );
        candidate.charged = charged;
        entries_.push_back(std::move(candidate));
        try {
            auto &entry = entries_.back();
            if (!arena_) arena_ = ops.allocate(ops.context, byte_limit_);
            if (!arena_) {
                entries_.pop_back();
                return PTO_RUNTIME_ERR_INTERNAL;
            }
            entry.residency.device_address = reinterpret_cast<uint64_t>(arena_) + used_;
            std::vector<uint8_t> scratch(entry.image);
            patch_chip_callable_scratch_for_device(callable, layout, entry.residency.device_address, scratch.data());
            rc = ops.copy(ops.context, reinterpret_cast<void *>(entry.residency.device_address), scratch.data(), bytes);
            if (rc != 0) {
                entries_.pop_back();
                return rc;
            }
            used_ += charged;
            out_callable_id = id;
        } catch (...) {
            entries_.pop_back();
            throw;
        }
        return 0;
    }

    void commit(int32_t id) {
        if (!entries_.empty() && entries_.back().residency.callable_id == id) entries_.back().ready = true;
    }
    void rollback(int32_t id) {
        if (!entries_.empty() && !entries_.back().ready && entries_.back().residency.callable_id == id) {
            used_ -= entries_.back().charged;
            entries_.pop_back();
        }
    }
    int resolve(int32_t id, KernelCallableResidency &out) const {
        out = {};
        if (id < 0 || id >= MAX_REGISTERED_CALLABLE_IDS) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (static_cast<size_t>(id) < entries_.size() && entries_[id].ready) {
            out = entries_[id].residency;
            return 0;
        }
        return PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT;
    }
    // The host-side registration bookkeeping runs after stage() uploaded the
    // image, and consumes that admission's upload. Committed residents are
    // never candidates, so an identical image already resident cannot be
    // mistaken for this one.
    uint64_t pending_uploaded_address() const {
        if (entries_.empty() || entries_.back().ready) return 0;
        return entries_.back().residency.device_address;
    }
    size_t resident_bytes() const { return used_; }
    size_t resident_count() const {
        return std::count_if(entries_.begin(), entries_.end(), [](const Entry &entry) {
            return entry.ready;
        });
    }
    size_t host_bytes() const { return used_ - padding_bytes(); }
    // MemoryAllocator owns the arena; this operation only drops host metadata.
    void clear() {
        entries_.clear();
        arena_ = nullptr;
        used_ = 0;
    }

    static int validate_image(const ChipCallable *callable, size_t bytes) {
        constexpr size_t function_capacity = std::extent_v<decltype(ChipCallable::child_func_ids_)>;
        if (!callable || bytes < sizeof(ChipCallable) || reinterpret_cast<uintptr_t>(callable) % alignof(ChipCallable))
            return PTO_RUNTIME_ERR_INTERNAL;
        if (callable->sig_count_ < 0 || callable->sig_count_ > CHIP_MAX_TENSOR_ARGS || callable->child_count_ < 0 ||
            static_cast<size_t>(callable->child_count_) > function_capacity ||
            callable->func_name_len_ >= CALLABLE_FUNC_NAME_MAX || callable->config_name_len_ >= CALLABLE_FUNC_NAME_MAX)
            return PTO_RUNTIME_ERR_INTERNAL;
        if (callable->func_name_[callable->func_name_len_] != '\0' ||
            callable->config_name_[callable->config_name_len_] != '\0')
            return PTO_RUNTIME_ERR_INTERNAL;
        int32_t scalars = 0;
        for (int32_t i = 0; i < callable->sig_count_; ++i) {
            const auto direction = callable->signature_[i];
            if (direction < ArgDirection::SCALAR || direction > ArgDirection::INOUT) return PTO_RUNTIME_ERR_INTERNAL;
            scalars += direction == ArgDirection::SCALAR;
        }
        if (scalars > CHIP_MAX_SCALAR_ARGS) return PTO_RUNTIME_ERR_INTERNAL;
        const size_t storage = bytes - offsetof(ChipCallable, storage_);
        size_t end = callable->binary_size_;
        if (end > storage) return PTO_RUNTIME_ERR_INTERNAL;
        for (int32_t i = 0; i < callable->child_count_; ++i) {
            const int32_t id = callable->child_func_ids_[i];
            const auto *ids_end = callable->child_func_ids_ + i;
            if (id < 0 || static_cast<size_t>(id) >= function_capacity ||
                std::find(callable->child_func_ids_, ids_end, id) != ids_end)
                return PTO_RUNTIME_ERR_INTERNAL;
            const size_t offset = callable->child_offsets_[i];
            if (offset % CALLABLE_ALIGN || offset < end || offset > storage ||
                CoreCallable::binary_data_offset() > storage - offset)
                return PTO_RUNTIME_ERR_INTERNAL;
            const auto &child = callable->child(i);
            if (child.sig_count_ < 0 || child.sig_count_ > CORE_MAX_TENSOR_ARGS ||
                child.binary_size_ > storage - offset - CoreCallable::binary_data_offset())
                return PTO_RUNTIME_ERR_INTERNAL;
            end = std::max(end, offset + CoreCallable::binary_data_offset() + child.binary_size_);
        }
        return end == storage ? 0 : PTO_RUNTIME_ERR_INTERNAL;
    }

private:
    struct Entry {
        KernelCallableResidency residency;
        std::vector<uint8_t> image;
        size_t charged{0};
        bool ready{false};
    };
    size_t padding_bytes() const {
        size_t padding = 0;
        for (const auto &entry : entries_)
            if (entry.charged) padding += entry.charged - entry.image.size();
        return padding;
    }
    size_t byte_limit_;
    size_t used_{0};
    void *arena_{nullptr};
    std::vector<Entry> entries_;
};
