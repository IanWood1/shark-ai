// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!q_type = tensor<{{b1}}x{{b2}}x{{l}}x{{d}}x{{i_dtype}}>
!mask_type = tensor<{{b1}}x{{b2}}x{{l}}x?x{{mask_dtype}}>
!o_type = tensor<{{b1}}x{{b2}}x{{l}}x{{d}}x{{o_dtype}}>
!cache_type = tensor<{{b1}}x{{block_stride}}x{{num_heads}}x{{head_dim}}x{{cache_dtype}}>
!gather_type = tensor<{{b1}}x?x{{block_stride}}x{{num_heads}}x{{head_dim}}x{{cache_dtype}}>
!page_ids_type = tensor<{{b1}}x?xi64>
!s_type = tensor<{{scale_dtype}}>

module {

util.func private @{{func_name}}(
    %q: !q_type,
    %s: !s_type,
    %mask: !mask_type,
    %cache: {{cache_tensor_type}},
    %page_ids: !page_ids_type
  ) -> !o_type {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %page_ids_dim = tensor.dim %page_ids, %c1 : !page_ids_type
  %page_ids_empty = tensor.empty(%page_ids_dim) : !page_ids_type
  %empty_gather = tensor.empty(%page_ids_dim) : !gather_type

  %block_index = arith.constant {{block_index}} : i64
  %page_stride = arith.constant {{page_stride}} : i64

  %key_indices = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%page_ids : !page_ids_type)
    outs(%page_ids_empty : !page_ids_type){
  ^bb0(%in : i64, %out : i64):
    %mul = arith.muli %in, %page_stride : i64
    %inc = arith.addi %mul, %block_index : i64
    linalg.yield %inc : i64
  } -> !page_ids_type

  %gathered_k = iree_linalg_ext.gather dimension_map = [0]
    ins(%cache , %key_indices : {{cache_tensor_type}}, !page_ids_type)
    outs(%empty_gather : !gather_type) {
    ^bb0(%arg0: {{cache_dtype}}, %arg1: {{cache_dtype}}):
      iree_linalg_ext.yield %arg0 : {{cache_dtype}}
  } -> !gather_type


  %value_indices = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%key_indices : !page_ids_type)
    outs(%page_ids_empty : !page_ids_type){
  ^bb0(%in : i64, %out : i64):
    %cst = arith.constant 1 : i64
    %res = arith.addi %in, %cst : i64
    linalg.yield %res : i64
  } -> !page_ids_type
  %gathered_v = iree_linalg_ext.gather dimension_map = [0]
                  ins(%cache , %value_indices : {{cache_tensor_type}}, !page_ids_type)
                  outs(%empty_gather : !gather_type) {
            ^bb0(%arg0: {{cache_dtype}}, %arg1: {{cache_dtype}}):
              iree_linalg_ext.yield %arg0 : {{cache_dtype}}
  } -> !gather_type


  %scale = tensor.extract %s[] : !s_type
  %empty = tensor.empty() : !o_type

  %attn = iree_linalg_ext.attention {indexing_maps = [
          affine_map<(d0, d1, d3, d4, d5, d6, d7) -> (d0, d1, d3, d5)>, 
          affine_map<(d0, d1, d3, d4, d5, d6, d7) -> (d0, d6, d7, d1, d5)>, 
          affine_map<(d0, d1, d3, d4, d5, d6, d7) -> (d0, d6, d7, d1, d4)>, 
          affine_map<(d0, d1, d3, d4, d5, d6, d7) -> ()>, 
          affine_map<(d0, d1, d3, d4, d5, d6, d7) -> (d0, d1, d6, d7)>, 
          affine_map<(d0, d1, d3, d4, d5, d6, d7) -> (d0, d1, d3, d4)>]}
    ins(%q, %gathered_k, %gathered_v, %scale, %mask : !q_type, !gather_type, !gather_type, {{scale_dtype}}, !mask_type) 
    outs(%empty : !o_type) {
  ^bb0(%score: {{o_dtype}}):
     iree_linalg_ext.yield %score : {{o_dtype}}
  } -> !o_type
  util.return %attn : !o_type
}

} // module
