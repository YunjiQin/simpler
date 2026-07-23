# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""simpler_setup.torch_interop.make_buffer_ref — the L3+ way to name a task arg (handle + torch view).

L3+ holds no C++ Tensor: a BufferHandle is the buffer, a torch tensor built over it (torch.frombuffer)
is for compute, and make_buffer_ref reads that torch tensor's geometry to build the wire BufferRef.
"""

import pytest
import torch
from _task_interface import DataType
from simpler.buffer_handle import BackendKind, create_host_shared_buffer, mint_owner_instance_id

from simpler_setup.torch_interop import make_buffer_ref


def _handle(nbytes=256):
    return create_host_shared_buffer(nbytes=nbytes, owner_instance_id=mint_owner_instance_id(), buffer_id=1)


def _view(handle, dtype, count, offset=0):
    shm = handle.shm
    assert shm is not None
    buf = shm.buf
    assert buf is not None
    return torch.frombuffer(buf, dtype=dtype, count=count, offset=offset)


def test_make_buffer_ref_geometry_from_torch():
    h = _handle()
    try:
        t = _view(h, torch.float32, 8)  # whole-buffer view, offset 0
        ref = make_buffer_ref(h, t)
        assert ref.handle == h.to_descriptor()
        assert ref.handle.backend_kind == BackendKind.POSIX_SHM
        assert ref.byte_offset == 0
        assert ref.shapes == (8,)
        assert ref.strides == (1,)
        assert ref.dtype == DataType.FLOAT32.value
    finally:
        h.close()


def test_make_buffer_ref_sub_view_byte_offset():
    h = _handle()
    try:
        t = _view(h, torch.float32, 4, offset=16)  # 16 bytes in
        ref = make_buffer_ref(h, t)
        assert ref.byte_offset == 16
        assert ref.shapes == (4,)
    finally:
        h.close()


def test_make_buffer_ref_rejects_non_contiguous():
    h = _handle()
    try:
        t = _view(h, torch.float32, 8)[::2]  # strided view
        with pytest.raises(ValueError, match="contiguous"):
            make_buffer_ref(h, t)
    finally:
        h.close()
