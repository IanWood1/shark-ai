# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Light-weight encapsulations for various forms of attention KV-caches.

These are not complete abstractions: they are primarily focused on making
tightly coupled transformer blocks a bit less "stringy" with loose tensors
and dims floating around everywhere.
"""

from typing import Optional, Tuple, Union, List
from sharktank.kernels.mlir_kernel import *
import abc
import math

import torch

from sharktank.types import (
    SplitPrimitiveTensor,
    ReplicatedTensor,
    QuantizerTensor,
    PlanarQuantizedTensor,
    StaticScaledQuantizer,
)
from sharktank import ops, kernels
from sharktank.kernels.mlir_kernel import *

__all__ = ["PagedAttention"]

# Paged Attention Kernels
#
# Each kernel is put into its own class to create a namespace for it


def KVCacheGatherKernel():
    CACHE_SIZE = DynDim.CACHE_SIZE
    PAGES = DynDim.PAGES
    BLOCK_SEQ_STRIDE = StaticDim.BLOCK_SEQ_STRIDE
    HEAD_COUNT_KV = StaticDim.HEAD_COUNT_KV
    ATTN_HEAD_DIM = StaticDim.ATTN_HEAD_DIM
    BATCH = DynDim.BATCH

    SOURCE_TY = Dtype.SOURCE_TY
    I64 = Dtype.I64

    @mlir_kernel(
        inputs=(
            MLIRTensor[
                CACHE_SIZE, BLOCK_SEQ_STRIDE, HEAD_COUNT_KV, ATTN_HEAD_DIM, SOURCE_TY
            ],
            MLIRTensor[BATCH, PAGES, I64],
        ),
        results=(
            MLIRTensor[
                BATCH, PAGES, BLOCK_SEQ_STRIDE, HEAD_COUNT_KV, ATTN_HEAD_DIM, SOURCE_TY
            ],
        ),
    )
    def paged_attention_kv_cache_gather(source, indices, result):
        # We generate the tensor.extract version for now, but once we have
        # iree_linalg_ext.gather, we should be generating that instead.
        mlir = """
        module {
        util.func @{{kernel_name}}(%source: !source, %indices: !indices) -> !result {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %batches = tensor.dim %indices, %c0 : !indices
          %pages = tensor.dim %indices, %c1 : !indices
          %empty = tensor.empty(%batches, %pages) : !result
          %result = linalg.generic {
            indexing_maps = [
            affine_map<(b, p, stride, head_count, head_dim) -> (b, p)>,
            affine_map<(b, p, stride, head_count, head_dim) -> (b, p, stride, head_count, head_dim)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
            ins(%indices : !indices)
            outs(%empty : !result) {
            ^bb0(%in: !indices_dtype, %o: !source_dtype):
              %p = arith.index_cast %in : !indices_dtype to index
              %stride = linalg.index 2 : index
              %head_count = linalg.index 3 : index
              %head_dim = linalg.index 4 : index
              %extracted = tensor.extract %source[%p, %stride, %head_count, %head_dim] : !source
              linalg.yield %extracted : !source_dtype
            } -> !result
            util.return %result : !result
        }
        }
        """
        return MLIRSpec(mlir)

    return paged_attention_kv_cache_gather


kv_cache_gather = KVCacheGatherKernel()

# Paged Attention Implementation


def PagedAttentionKernel():
    GQA_REP = StaticDim.GQA_REP
    BLOCK_SEQ_STRIDE = StaticDim.BLOCK_SEQ_STRIDE
    HEAD_COUNT_KV = StaticDim.HEAD_COUNT_KV
    BLOCK_SEQ_LEN = DynDim.BLOCK_SEQ_LEN
    SEQ_LEN = DynDim.SEQ_LEN
    QEMB = StaticDim.QEMB
    HEAD_DIM = StaticDim.HED_DIM
    BATCH = StaticDim.BATCH
    UNIT = StaticDim.SCALE

    SOURCE_TY = Dtype.SOURCE_TY

    # TODO: scale is casted to f32 so it is used to make the output f32.
    # Remove when specific output dtypes can be specified.
    SCALE_TY = Dtype.SCALE_TY
    MASK_TY = Dtype.MASK_TY
    RESULT_TY = SOURCE_TY

    @mlir_kernel(
        inputs=(
            MLIRTensor[BATCH, SEQ_LEN, HEAD_COUNT_KV, GQA_REP, QEMB, SOURCE_TY],
            MLIRTensor[
                BATCH,
                BLOCK_SEQ_LEN,
                BLOCK_SEQ_STRIDE,
                HEAD_COUNT_KV,
                HEAD_DIM,
                SOURCE_TY,
            ],
            MLIRTensor[
                BATCH,
                BLOCK_SEQ_LEN,
                BLOCK_SEQ_STRIDE,
                HEAD_COUNT_KV,
                HEAD_DIM,
                SOURCE_TY,
            ],
            MLIRTensor[UNIT, SCALE_TY],
            MLIRTensor[BATCH, SEQ_LEN, BLOCK_SEQ_LEN, BLOCK_SEQ_STRIDE, MASK_TY],
        ),
        results=(
            MLIRTensor[BATCH, HEAD_COUNT_KV, GQA_REP, SEQ_LEN, HEAD_DIM, SCALE_TY],
        ),
    )
    def paged_attention_kernel(q, k, v, scale, mask, result):
        # We generate the tensor.extract version for now, but once we have
        # iree_linalg_ext.gather, we should be generating that instead.
        mlir = """
        module {
        util.func @{{kernel_name}}(
            %q: !q, %k: !k, %v : !v, %scale : !scale, %mask : !mask
        ) -> !result {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %scale_scalar = tensor.extract %scale[%c0] : !scale
          %seq_len = tensor.dim %q, %c1 : !q
          %empty = tensor.empty(%seq_len) : !result
          %result = iree_linalg_ext.attention {indexing_maps = [
                affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3, d1, d2, d5)>, 
                affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d6, d7, d1, d5)>, 
                affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d6, d7, d1, d4)>, 
                affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>, 
                affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3, d6, d7)>, 
                affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>]}
            ins(%q, %k, %v, %scale_scalar, %mask : !q, !k, !v, f32, !mask) 
            outs(%empty : !result) {
          ^bb0(%score: f32):
             iree_linalg_ext.yield %score : f32
          } -> !result
          util.return %result : !result
        }
        }
        """
        return MLIRSpec(mlir)

    return paged_attention_kernel


_paged_attention_kernel = PagedAttentionKernel()


def paged_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    scale: torch.Tensor,
):
    bs, sl, num_heads, head_dim = q.shape
    _, block_seq_len, block_seq_stride, kv_heads, _ = k.shape

    if mask is None:
        mask = torch.full(
            (sl, block_seq_len * block_seq_stride), float("-inf"), dtype=q.dtype
        )
        mask = torch.triu(mask, diagonal=1)
    mask = mask[None, :, :]
    # mask = ops.expand(mask, (bs, sl, block_seq_len * block_seq_stride))
    mask = mask.reshape(bs, sl, block_seq_len, block_seq_stride)
    print(q.dtype)
    print(mask.shape)
    print(mask)

    print(f"qshape {q.shape}")
    q = q.reshape(bs, sl, kv_heads, num_heads // kv_heads, head_dim)
    print(f"qshape {q.shape}")

    return _paged_attention_kernel(
        q=q,
        k=k,
        v=v,
        mask=mask,
        scale=scale.to(dtype=torch.float32),
    ).flatten(1, 2)


class PagedAttention:
    """Implementation of paged attention

    The page table slab is physically represented as a 2D tensor:
        [page_count, flattened_dims]

    Each "page" can be thought of as a 6D view onto:

    * transformer block
    * cache partition (K or V cache)
    * block sequence stride (number of sequence positions per block)
    * attention heads
    * attention dimensionality

    Note that the internal page structure matches the organization of the
    model, allowing contiguous individual local reads and writes at a sub-block
    granularity if indexing deeply into the structure.

    When `shard_count > 1`, it would split the `attn_head_count` dimension.
    The page slab is a 1D sharded split tensor.
    It is reinterpreted as a 6D tensor, by working around the lack of sharded
    block-cyclic sharded tensor type.
    """

    def __init__(
        self,
        *,
        transformer_block_count: int,
        attn_head_count: int,
        attn_head_dim: int,
        cache_partition_count: int = 2,
        block_seq_stride: int = 16,
        cache_dtype: torch.dtype = torch.float32,
        attn_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        shard_count: int = 1,
        block_to_device_lookup: tuple[tuple[int, ...], ...] | None = None,
    ):
        self.transformer_block_count = transformer_block_count
        self.head_count_kv = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.cache_partition_count = cache_partition_count
        self.block_seq_stride = block_seq_stride
        self.shard_count = shard_count

        if block_to_device_lookup is None:
            block_to_device_lookup = tuple(
                tuple(range(self.shard_count))
                for _ in range(self.transformer_block_count)
            )
        assert len(block_to_device_lookup) == transformer_block_count
        block_to_pipeline_lookup = [0]
        pipeline_to_device_lookup = [block_to_device_lookup[0]]
        pipeline = 0
        for block in range(1, transformer_block_count):
            ds_prev, ds_curr = (
                block_to_device_lookup[block - 1],
                block_to_device_lookup[block],
            )
            assert all(d for d in ds_prev) >= 0
            assert all(d for d in ds_curr) >= 0
            if not all(d_prev == d_curr for d_prev, d_curr in zip(ds_prev, ds_curr)):
                pipeline += 1
                pipeline_to_device_lookup.append(ds_curr)
            block_to_pipeline_lookup.append(pipeline)

        self.pipeline_to_device_lookup = tuple(pipeline_to_device_lookup)
        self.block_to_pipeline_lookup = tuple(block_to_pipeline_lookup)
        self.pipeline_count = len(pipeline_to_device_lookup)

        if attn_head_count % shard_count != 0:
            raise ValueError(
                f"The attention head count {attn_head_count} must be a multiple of the tensor parallelism size {shard_count}."
            )

        self.pipeline_to_block_count = tuple(
            sum(1 for block in block_to_pipeline_lookup if block == i)
            for i in range(self.pipeline_count)
        )
        # Some derived values based on attributes.
        self.sub_page_dims = [
            [
                self.pipeline_to_block_count[pipeline],
                self.cache_partition_count,
                self.block_seq_stride,
                self.head_count_kv // self.shard_count,
                self.attn_head_dim,
            ]
            for pipeline in range(self.pipeline_count)
        ]
        self.page_slab_flat_dims = [
            math.prod(sub_page_dim) for sub_page_dim in self.sub_page_dims
        ]
        self.device = device
        self.cache_dtype = cache_dtype
        self.attn_dtype = attn_dtype

    def unflatten_page_tables(
        self, state: list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor]
    ) -> list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor]:
        """Unflattens the 2D page tables to 6D tensors."""
        assert (
            len(state) == self.pipeline_count
        ), f"Expected {self.pipeline_count}-element state. Got: {len(state)}"

        if self.shard_count == 1 and self.pipeline_count == 1:
            assert all(
                not isinstance(page_slab, SplitPrimitiveTensor) for page_slab in state
            )
            return [
                page_slab.unflatten(1, self.sub_page_dims[pipeline])
                for pipeline, page_slab in enumerate(state)
            ]

        assert all(page_slab.shard_count == self.shard_count for page_slab in state)
        unflattened = []
        for pipeline, page_slab in enumerate(state):
            shards = [
                shard.unflatten(1, self.sub_page_dims[pipeline])
                for shard in page_slab.shards
            ]
            unflattened.append(
                (
                    SplitPrimitiveTensor(
                        ts=shards, shard_dim=4, devices=page_slab.devices
                    )
                    if len(shards) > 1
                    else ReplicatedTensor(ts=shards, devices=page_slab.devices)
                )
            )
        return unflattened

    def shard_state(
        self, state: List[torch.Tensor]
    ) -> List[torch.Tensor | SplitPrimitiveTensor]:
        """Shard an unsharded state.
        We can't just split the slab on the sub page dims.
        First it needs to be reinterpreted into the actual shape.
        The split the head dimension, then flatten each shard.
        This is a work-around for the lack of block-cyclic sharded tensor type."""
        if self.shard_count == 1 and self.pipeline_count == 1:
            return state

        page_table = state[0].reshape(
            [
                -1,
                self.transformer_block_count,
                self.cache_partition_count,
                self.block_seq_stride,
                self.head_count_kv,
                self.attn_head_dim,
            ]
        )

        flat_sharded_page_tables = []
        for pipeline in range(self.pipeline_count):
            # TODO: Do I need to make copies here, or are views enough?
            assert (
                self.pipeline_to_block_count[pipeline] != 1
            ), "1 tensor per pipeline not supported. dim gets collapsed"
            i_min = sum(self.pipeline_to_block_count[:pipeline])
            i_max = i_min + self.pipeline_to_block_count[pipeline]
            if self.shard_count == 1:
                sharded_page_table = ops.replicate(
                    page_table[:, i_min:i_max, ...],
                    count=1,
                    devices=self.pipeline_to_device_lookup[pipeline],
                )
            else:
                sharded_page_table = ops.reshard_split(
                    page_table[:, i_min:i_max, ...],
                    dim=4,
                    count=self.shard_count,
                    devices=self.pipeline_to_device_lookup[pipeline],
                )
            shards_flattened = [
                ops.flatten(shard, start_dim=1) for shard in sharded_page_table.shards
            ]
            flat_sharded_page_tables.append(
                SplitPrimitiveTensor(
                    ts=shards_flattened,
                    shard_dim=1,
                    devices=self.pipeline_to_device_lookup[pipeline],
                )
                if self.shard_count > 1
                else ReplicatedTensor(
                    ts=shards_flattened,
                    devices=self.pipeline_to_device_lookup[pipeline],
                )
            )
        return flat_sharded_page_tables

    @property
    def pad_sequence_stride(self) -> int:
        return self.block_seq_stride

    def allocate(
        self, page_count: int
    ) -> list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor]:
        """Allocates tensor state for a page table for the given capacity in
        pages.
        """
        shards = [
            [
                torch.empty(
                    [page_count, self.page_slab_flat_dims[pipeline]],
                    dtype=self.cache_dtype,
                    device=self.device,
                )
                for _ in range(self.shard_count)
            ]
            for pipeline in range(self.pipeline_count)
        ]

        if self.shard_count == 1 and self.pipeline_count == 1:
            return shards[0]

        return [
            (
                SplitPrimitiveTensor(ts=shards[i], shard_dim=1, devices=devices)
                if len(shards[i]) > 1
                else ReplicatedTensor(ts=shards[i], devices=devices)
            )
            for i, devices in enumerate(self.pipeline_to_device_lookup)
        ]

    def read(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        seq_len: int,
        page_ids: Optional[Union[torch.Tensor, ReplicatedTensor]] = None,
    ):
        """Reads K/V caches the page table for the given page_ids.

        Args:
        state: State struct as returned from allocate().
        transformer_block_index: The index of the transformer block accessing
            the cache.
        page_ids: Tensor of [bs, max_seqlen // block_pos_stride] of page ids
            to access.

        Returns the K/V cache partitions, linearized. Note that this reference
        approach to reading by materializing linearly may not be terribly
        efficient unless if the compiler can fuse the gather.
        """
        page_tables = self.unflatten_page_tables(state)  # 6D
        page_table = page_tables[self.block_to_pipeline_lookup[transformer_block_index]]

        # TODO: IREE codegen cannot handle strided gathers properly today.
        # vector.gather codegen for IREE is simply broken for strided gathers.
        # This is sad, but to work around it, we flatten the strides into the
        # gather indices. This is okay, because we aren't exactly linearizing
        # two indices vectors into one, but only adding strides to it.
        #
        # The page table layout is organized as:
        #   [page_id, attn_layer, cache_partition, page]
        # Where the cache line can be 0 (k) or 1 (v).
        #
        # For a particular attention layer, attn_layer and cache_partition are
        # fixed, and we are gathering over the page_id dimension. We flatten
        # page_id, attn_layer, cache_partition into a single dimension, and
        # gather over it, allowing us to bypass IREE's strided gather codegen.

        # Get strided k/v page ids.
        k_strided_page_ids = (
            (page_ids * self.transformer_block_count * self.cache_partition_count)
            + (transformer_block_index * self.cache_partition_count)
            + 0
        )
        v_strided_page_ids = (
            (page_ids * self.transformer_block_count * self.cache_partition_count)
            + (transformer_block_index * self.cache_partition_count)
            + 1
        )
        strided_page_table = page_table.flatten(0, 2)

        # In another world where torch.tensor __getitem__ did not generate
        # negative indexing checks, we would directly index the tensor, which
        # is a much better way of indexing into tensors. Maybe one day we will
        # have a range analysis to remove those checks. But today, we use
        # inline mlir to generate a gather without the negative indexing checks.
        key = kv_cache_gather(strided_page_table, k_strided_page_ids)
        value = kv_cache_gather(strided_page_table, v_strided_page_ids)

        return key, value

    def write_timestep(
        self,
        state: list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        # List of [bs, 1, attn_head_count, attn_head_dim]
        cache_partitions: list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        *,
        transformer_block_index: int,
        # [bs]
        seq_positions: Union[torch.Tensor, ReplicatedTensor],
        # [bs, max_seqlen // block_pos_stride]
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes a single batched timestep across all cache partitions.

        Note that this internally loops over the batch size, which cannot be
        dynamic.
        """
        device = self.device
        page_tables = self.unflatten_page_tables(state)  # 6D
        page_table = page_tables[self.block_to_pipeline_lookup[transformer_block_index]]
        page_table = page_table.flatten(0, 3)
        bs, *_ = seq_positions.shape
        assert len(cache_partitions) == self.cache_partition_count

        # [bs, 1, atten_head_count, attn_head_dim]
        for idx, cache_partition in enumerate(cache_partitions):
            # [bs, 1]
            page_index = seq_positions // self.block_seq_stride

            page_id = ops.gather(page_ids, dim=1, index=page_index.unsqueeze(1))
            page_offset = (seq_positions % self.block_seq_stride).unsqueeze(1)

            # [1, 1]
            if isinstance(seq_positions, ReplicatedTensor):
                partitions = [
                    torch.tensor(idx, device=device).unsqueeze(0)
                    for _ in range(seq_positions.shard_count)
                ]

                transformer_block = [
                    torch.full((bs, 1), transformer_block_index, device=device)
                    for _ in range(seq_positions.shard_count)
                ]

                devices = self.pipeline_to_device_lookup[
                    self.block_to_pipeline_lookup[transformer_block_index]
                ]
                partitions = ReplicatedTensor(ts=partitions, devices=devices)
                transformer_block = ReplicatedTensor(
                    ts=transformer_block, devices=devices
                )
            else:
                partitions = torch.tensor(idx, device=device).unsqueeze(0)
                transformer_block = torch.full(
                    (bs, 1), transformer_block_index, device=device
                )

            partitions = partitions.repeat(bs, 1)

            transformer_block_count_in_pipeline = self.pipeline_to_block_count[
                self.block_to_pipeline_lookup[transformer_block_index]
            ]

            index = page_id
            index = index * transformer_block_count_in_pipeline + transformer_block
            index = index * self.cache_partition_count + partitions
            index = index * self.block_seq_stride + page_offset
            values = ops.to(cache_partition, dtype=page_table.dtype)
            if page_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                page_table_as_int8 = page_table.view(dtype=torch.int8)
                values_int8 = values.view(dtype=torch.int8)
                page_table_as_int8.index_put_(indices=(index,), values=values_int8)
            else:
                page_table.index_put_(indices=(index,), values=values)

        return

    def write(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        cache_partitions: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes cache partitions from a linear layout to the page table.

        This is the inverse of the linear read. The same caveat applies if the
        in-place scatter cannot be fused.
        """
        page_tables = self.unflatten_page_tables(state)  # 6D
        page_table = page_tables[self.block_to_pipeline_lookup[transformer_block_index]]
        bs, block_seq_len, *_ = page_ids.shape

        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache partition.
        # This requires us to recompute indices to the sub-block reference
        # frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_partition]
        # Where the cache line can be 0 (k) or 1 (v).
        subblock_table = page_table.flatten(start_dim=0, end_dim=2)
        transformer_block_count_in_pipeline = self.pipeline_to_block_count[
            self.block_to_pipeline_lookup[transformer_block_index]
        ]
        page_stride = transformer_block_count_in_pipeline * self.cache_partition_count
        transformer_block_stride = self.cache_partition_count
        base_subblock_ids = page_ids * page_stride + (
            transformer_block_index * transformer_block_stride
        )

        for index, partition in enumerate(cache_partitions):
            part_block_view = partition.flatten(0, 1)

            subblock_ids = (
                (base_subblock_ids + index) if index > 0 else base_subblock_ids
            ).flatten(0, 1)

            part_block = ops.to(part_block_view, dtype=subblock_table.dtype)
            if subblock_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                subblock_table_as_int8 = subblock_table.view(dtype=torch.int8)
                part_block_as_int8 = part_block.view(dtype=torch.int8)
                subblock_table_as_int8.index_copy_(0, subblock_ids, part_block_as_int8)
            else:
                subblock_table.index_copy_(0, subblock_ids, part_block)

    def attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_count_attn: int,
        attention_kernel: str,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        probs_quantizer: Optional[StaticScaledQuantizer] = None,
    ):
        gqa_n_rep = head_count_attn // self.head_count_kv
        assert gqa_n_rep > 0

        def repeat_kv(x: torch.Tensor) -> torch.Tensor:
            bs, slen, n_kv_heads, head_dim = x.shape
            unsq = x.unsqueeze(-2)
            exp = ops.expand(unsq, (bs, slen, n_kv_heads, gqa_n_rep, head_dim))
            return exp.flatten(2, 3)

        # if gqa_n_rep > 1:
        #     k = repeat_kv(k)
        #     v = repeat_kv(v)
        #
        # # Fake quant is already dequantized when stored in the cache.
        # if cache_quantizer and not fake_quant:
        #     k = cache_quantizer.dequantize_raw_tensor(k, self.attn_dtype, name="xk_deq")
        #     v = cache_quantizer.dequantize_raw_tensor(v, self.attn_dtype, name="xv_deq")
        #
        # q = q.transpose(1, 2)
        # k = k.transpose(1, 2)
        # v = v.transpose(1, 2)
        #
        q = ops.to(q, dtype=self.attn_dtype)
        k = ops.to(k, dtype=self.attn_dtype)
        v = ops.to(v, dtype=self.attn_dtype)
        if mask is not None:
            mask = ops.to(mask, dtype=self.attn_dtype)

        # Decomposed
        if attention_kernel == "decomposed":
            if isinstance(q, PlanarQuantizedTensor):
                q = q.unpack().dequantize()
            if isinstance(k, PlanarQuantizedTensor):
                k = k.unpack().dequantize()
            if isinstance(v, PlanarQuantizedTensor):
                v = v.unpack().dequantize()

            attn_weights = ops.matmul(
                q.to(torch.float32), k.transpose(2, 3).to(torch.float32)
            )
            attn_weights = attn_weights / math.sqrt(self.attn_head_dim)

            # Flash attention.
            if softcap is not None:
                attn_weights = softcap * torch.tanh(attn_weights / softcap)

            # Apply attention mask.
            if mask is None:
                mask = torch.full(
                    (attn_weights.shape[2], attn_weights.shape[3]), float("-inf")
                )
                mask = torch.triu(mask, diagonal=1)[None, None, :, :]
                attn_weights = attn_weights + mask
            else:
                attn_weights = attn_weights + mask

            attn_weights = ops.softmax(
                ops.to(attn_weights, dtype=torch.float32), dim=-1
            )
            if probs_quantizer is not None:
                if fake_quant:
                    attn_weights = (
                        probs_quantizer.quantize(attn_weights).unpack().dequant()
                    )
                else:
                    attn_weights = probs_quantizer.quantize(attn_weights).unpack().qs
            attn_weights = ops.to(attn_weights, dtype=q.dtype)
            return ops.matmul(attn_weights, v)  # (bs, heads, slen, head_dim)
        elif attention_kernel == "sharktank":
            if mask is not None:
                attn_output = kernels.masked_flash_attention(
                    q,
                    k,
                    v,
                    mask[0, 0, :, :],
                    torch.tensor(1 / math.sqrt(self.attn_head_dim)),
                )
            else:
                attn_output = kernels.flash_attention(q, k, v)
            return attn_output
        else:
            # Non-decomposed
            if softcap is not None:
                raise ValueError("softcap not supported yet")

            # TODO
            assert not (cache_quantizer and not fake_quant)

            return paged_attention_kernel(
                q=q,  # [bs, ..., sl, dim]
                k=k,  # [bs, ..., sl, dim]
                v=v,  # [bs, ..., sl, dim]
                mask=mask,  # [bs, ..., sl, sl]
                # is_causal=mask is None,  # assumes causal masking when true
                scale=torch.tensor(
                    [1 / math.sqrt(self.attn_head_dim)]
                ),  # defaults to 1/sqrt(dim)
            )

    def forward_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_state: list[torch.Tensor],
        seq_block_ids: torch.Tensor,
        kv_seq_len: int,
        block_index: int,
        start_positions: torch.Tensor,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        # Write our one updated cache row into the cache.
        self.write_timestep(
            cache_state,
            cache_partitions=[
                k,
                v,
            ],
            transformer_block_index=block_index,
            seq_positions=start_positions,
            page_ids=seq_block_ids,
        )

        # Restore from the cache.
        k, v = self.read(
            cache_state,
            transformer_block_index=block_index,
            page_ids=seq_block_ids,
            seq_len=kv_seq_len,
        )

        return self.attention(
            q=q,
            k=k,
            v=v,
            head_count_attn=head_count_attn,
            attention_kernel=attention_kernel,
            cache_quantizer=cache_quantizer,
            fake_quant=fake_quant,
            softcap=softcap,
            scale=scale,
            mask=mask,
        )

    def forward_prefill(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_state: list[torch.Tensor],
        seq_block_ids: torch.Tensor,
        block_index: int,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        probs_quantizer: Optional[StaticScaledQuantizer] = None,
    ):
        _, block_seq_len, *_ = seq_block_ids.shape
        k = k.unflatten(1, (block_seq_len, self.block_seq_stride))
        v = v.unflatten(1, (block_seq_len, self.block_seq_stride))

        self.write(
            cache_state,
            cache_partitions=[k, v],
            transformer_block_index=block_index,
            page_ids=seq_block_ids,
        )

        return self.attention(
            q=q,
            k=k,
            v=v,
            head_count_attn=head_count_attn,
            attention_kernel=attention_kernel,
            cache_quantizer=cache_quantizer,
            fake_quant=fake_quant,
            softcap=softcap,
            scale=scale,
            mask=mask,
            probs_quantizer=probs_quantizer,
        )
