// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!input_tensor_type = {{input_tensor_type}}
!table_tensor_type = {{table_tensor_type}}
!reshaped_tensor_type = tensor<{{bs}}x{{"?" if sl == "D" else sl}}x{{heads}}x{{dims//2}}x2x{{dtype}}>


module {

util.func private @sharktank_rotary_embedding_{{bs}}_{{sl}}_{{heads}}_{{dims}}_{{dtype}}(%input: !input_tensor_type, %table: !table_tensor_type) -> !input_tensor_type {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index


  %d0 = tensor.dim %input, %c0 : !input_tensor_type
  %d1 = tensor.dim %input, %c1 : !input_tensor_type
  %d2 = tensor.dim %input, %c2 : !input_tensor_type
  %d3 = tensor.dim %input, %c3 : !input_tensor_type

  %empty_dyn = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?x{{dtype}}>
  %empty = tensor.cast %empty_dyn : tensor<?x?x?x?x{{dtype}}> to {{input_tensor_type}}
  %cst = arith.constant 0.000000e+00 : {{dtype}}
  %fill = linalg.fill ins(%cst : {{dtype}}) outs(%empty : !input_tensor_type) -> !input_tensor_type

  %expanded = tensor.expand_shape %input [[0], [1], [2], [3, 4]]
    output_shape[{{bs}}, {{"%d1" if sl == "D" else sl}}, {{heads}}, {{dims//2}}, 2]
    : !input_tensor_type into !reshaped_tensor_type

  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 floordiv 2, d4)>,
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%table, %expanded : !table_tensor_type, !reshaped_tensor_type)
    outs(%fill : !input_tensor_type) {
  ^bb0(%in: {{dtype}}, %in_0: {{dtype}}, %out: {{dtype}}):
    %3 = linalg.index 0 : index
    %4 = linalg.index 1 : index
    %5 = linalg.index 2 : index
    %6 = linalg.index 3 : index
    %7 = linalg.index 4 : index

    %8 = math.cos %in : {{dtype}}
    %9 = arith.mulf %in_0, %8 : {{dtype}}

    %10 = math.sin %in : {{dtype}}
    %11 = arith.mulf %in_0, %10 : {{dtype}}

    %12 = arith.negf %11 : {{dtype}}

    //%13 = arith.divui %6, %c2 : index
    %14 = arith.remui %6, %c2 : index
    %15 = arith.cmpi eq, %14, %c0 : index
    %16 = arith.select %15, %12, %11 : {{dtype}}
    %17 = arith.cmpi eq, %7, %c0 : index


    %18 = arith.select %17, %9, %16 : {{dtype}}
    %19 = arith.addf %out, %18 : {{dtype}}
    linalg.yield %19 : {{dtype}}
  } -> !input_tensor_type
  util.return %result : !input_tensor_type
}

}
