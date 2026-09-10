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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernel_invocation_header.h"

// CANN copies this prefix and the following payload into each launch snapshot.
// The binder supplies packet_bytes from the actual allocation size. This is
// a kernel invocation envelope, not the program entry's KernelArgs layout.
struct SimplerKernelDispatchArgs {
    uint64_t packet_bytes;
    // Filled from the issuing context's committed residency, not a Host image.
    // The immutable device image outlives every referencing launch and graph.
    uint64_t chip_callable_address;
    uint64_t chip_callable_bytes;
    SimplerKernelInvocationHeader invocation;
};

// Nonzero device-entry status is surfaced by caller synchronization.
enum class KernelDispatchStatus : int32_t {
    Success = 0,
    InvalidArgs = 1,
    UnsupportedPayload = 4,
};

static_assert(
    std::is_trivially_copyable_v<SimplerKernelDispatchArgs> && std::is_standard_layout_v<SimplerKernelDispatchArgs>
);
static_assert(sizeof(SimplerKernelDispatchArgs) == 56);
static_assert(offsetof(SimplerKernelDispatchArgs, chip_callable_address) == 8);
static_assert(offsetof(SimplerKernelDispatchArgs, chip_callable_bytes) == 16);
static_assert(offsetof(SimplerKernelDispatchArgs, invocation) == 24);

extern "C" int simpler_aicpu_kernel_exec(void *args);
