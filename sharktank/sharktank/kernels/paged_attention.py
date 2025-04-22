# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import cast

from iree.turbine.runtime.op_reg.base import AttrArg
from sharktank.kernels.base import *

import torch

__all__ = [
    "paged_attention",
]


@CustomOp.register(library=LIBRARY)
class paged_attention(CustomOp):

    signature = "paged_attention(Tensor q, Tensor scale, Tensor mask, Tensor cache, int block_index, int page_stride, Tensor page_ids) -> (Tensor)"

    def select(self, sel: KernelSelection):
        q_desc = sel.arg_tensor(0)  # Shape b, l, d
        scale_desc = sel.arg_tensor(1)
        mask_desc = sel.arg_tensor(2)
        cache_desc = sel.arg_tensor(3)
        block_index_desc = sel.attr_int(4)
        page_stride = sel.attr_int(5)
        page_ids_desc = sel.arg_tensor(6)

        # Note: kernel does collapse dims to get to a single batch/head dim
        q_bs = q_desc.t.shape[:-2]
        torch._check(len(q_bs) == 2, lambda: f"TODO: batch dims {q_bs} not supported")

        q_l, q_d = q_desc.t.shape[-2:]

        q_desc.specialize_all_dims()
        cache_desc.specialize_dims(1, 2, 3)
        page_ids_desc.specialize_dims(0)
        mask_desc.specialize_dims(0, 1, 2)

        sel.return_new_tensor(q_desc.t.shape, dtype=torch.float32).specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        q = kb.arg_value(0)  # Shape b, l, d
        scale = kb.arg_value(1)
        mask = kb.arg_value(2)
        cache = kb.arg_value(3)
        block_index = cast(AttrArg, ksel.arg_descs[4]).v
        page_stride = cast(AttrArg, ksel.arg_descs[5]).v
        page_ids = kb.arg_value(6)

        q_tensor_type = RankedTensorType(q.type)
        cache_tensor_type = RankedTensorType(cache.type)
        scale_tensor_type = RankedTensorType(scale.type)
        mask_tensor_type = RankedTensorType(mask.type)
        page_ids_tensor_type = RankedTensorType(page_ids.type)

        b1, b2, l, d = q_tensor_type.shape
        _, s = page_ids_tensor_type.shape
        _, block_stride, num_heads, head_dim = cache_tensor_type.shape

        # Unspecialized dims will be negative
        l = l if l >= 0 else "?"
        s = s if s >= 0 else "?"
        b = str(int(b1) * int(b2))
        i_type_str = str(q_tensor_type.element_type)
        scale_type_str = str(scale_tensor_type.element_type)
        mask_type_str = str(mask_tensor_type.element_type)
        # TODO: enable f16 output type via arg
        o_type_str = "f32"

        target_function_name = f"sharktank_paged_attention_{b1}_{b2}_{d}_{i_type_str}_{mask_type_str}_{scale_type_str}_{o_type_str}"
        kwargs = {
            "b": b,
            "b1": b1,
            "b2": b2,
            "l": l,
            "d": d,
            "s": s,
            "block_index": block_index,
            "cache_tensor_type": cache_tensor_type,
            "page_stride": page_stride,
            "block_stride": block_stride,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "mask_dtype": mask_type_str,
            "i_dtype": i_type_str,
            "cache_dtype": cache_tensor_type.element_type,
            "scale_dtype": scale_type_str,
            "o_dtype": o_type_str,
            "func_name": target_function_name,
        }
        template_file = "paged_attention.mlir"
        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            **kwargs,
        )
        kb.yield_results(
            *call_function(
                target_function,
                q,
                scale,
                mask,
                cache,
                page_ids,
            )
        )
        pass
