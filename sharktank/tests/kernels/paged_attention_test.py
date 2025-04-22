# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest
from parameterized import parameterized

import torch

from iree.turbine import aot
from sharktank import kernels
from sharktank import ops


class custom_paged_attention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(420)

    @parameterized.expand(
        [
            (torch.float32, 5e-3, 1e-3, True),
            (torch.float16, 5e-3, 1e-3, True),
            (torch.float32, 5e-3, 1e-3, False),
            (torch.float16, 5e-3, 1e-3, False),
        ]
    )
    def test_compare_torch_spda(self, dtype, atol, rtol, use_mask):
        num_heads = 10
        bs = 4
        in_seq_len = 1
        out_seq_len = 4
        Eqk = Ev = 64
        block_stride = 32
        transformer_count = 8
        num_part = 2
        block_index = 5
        num_pages = bs * out_seq_len * num_part * transformer_count

        q = torch.rand([bs, num_heads, in_seq_len, Eqk], dtype=dtype)
        page_ids = torch.reshape(torch.arange(bs * out_seq_len), (bs, out_seq_len))
        cache = torch.rand(
            [num_pages, block_stride, num_heads, Ev],
            dtype=dtype,
        )
        mask = torch.zeros(
            [bs, num_heads, in_seq_len, out_seq_len * block_stride], dtype=dtype
        )
        scale = torch.tensor(2.0, dtype=dtype)
        if use_mask:
            range_vector = torch.arange(0, out_seq_len * block_stride, 1)
            mask = range_vector >= out_seq_len
            mask = (
                mask.unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(bs, num_heads, in_seq_len, 1)
            )

        res2 = kernels.paged_attention(
            q=q,
            scale=scale,
            mask=mask,
            cache=cache,
            block_index=block_index,
            page_stride=transformer_count,
            page_ids=page_ids,
        )

        subblock_ids = page_ids * transformer_count + block_index
        subblock_ids_shape = subblock_ids.shape
        k = ops.index_select(cache, 0, subblock_ids.flatten(0, 1))
        v = ops.index_select(cache, 0, subblock_ids.flatten(0, 1) + 1)
        k = k.unflatten(0, subblock_ids_shape).flatten(1, 2).transpose(1, 2)
        v = v.unflatten(0, subblock_ids_shape).flatten(1, 2).transpose(1, 2)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, mask, scale=scale
        )

        torch.testing.assert_close(res2.to(dtype), ref, atol=atol, rtol=rtol)
