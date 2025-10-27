// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the inline definitions for all the MLIR assembly
// generation methods on the `Graph`, `TensorAttr`, `INode` and derived node
// classes. It is meant to be a common place for all things ASM emitter related
// to make maintenance and future improvements easier.
//
// We use a combination of raw multi-line strings `R"(...)"` and `std::format`
// (from C++20) to implement a simple templating system for generating MLIR
// assembly code. This could be made better with a jinja2-like templating
// system but for now this gets us mostly what we need.
//
// Caution: An important foot-gun with `std::format` is to forget to double the
// brace for a literal `{` or `}`. i.e. always use `{{` for `{` and `}}` for `}`
// to disambiguate from the `{}` that `std::format` uses for replacements.
// If not you'll hit a compilation error like so:
//    "error: call to consteval function 'std::basic_format_string<char, ...'"
//    "is not a constant expression"
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_ASM_EMITTER_H
#define FUSILLI_SUPPORT_ASM_EMITTER_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/attributes/types.h"
#include "fusilli/graph/graph.h"
#include "fusilli/node/conv_node.h"
#include "fusilli/support/extras.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format> // C++20
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fusilli {

// Given a vector of ints, returns the MLIR assembly for the
// `torch.constant.int` ops for each int value and the
// `torch.prim.ListConstruct` op wrapping these into a single
// value.
//
// For example if `getListOfIntOpsAsm` is called on these inputs:
//    listOfInts: {1, 2}
//    prefix: "stride"
//    suffix: "conv"
//
// It generates the following MLIR assembly:
//
//   %stride_val_0_conv = torch.constant.int 1
//   %stride_val_1_conv = torch.constant.int 2
//   %stride_conv = torch.prim.ListConstruct
//          %stride_val_0_conv, %stride_val_1_conv :
//              (!torch.int, !torch.int) -> !torch.list<int>
//
// The prefix is generally what attribute this refers to (e.g.
// padding, stride, dilation etc.) and the suffix is the node's
// unique name (for SSA disambiguation).
inline std::string getListOfIntOpsAsm(const std::vector<int64_t> &listOfInts,
                                      const std::string &prefix,
                                      const std::string &suffix) {
  std::ostringstream oss;
  std::vector<std::string> ssaValueNames;

  // Emit `torch.constant.int` ops for each int value.
  for (size_t i = 0; i < listOfInts.size(); ++i) {
    const std::string ssaValueName =
        "%" + prefix + "_val_" + std::to_string(i) + "_" + suffix;
    oss << ssaValueName << " = torch.constant.int " << listOfInts[i]
        << "\n    ";
    ssaValueNames.push_back(ssaValueName);
  }

  // Emit the ListConstruct op.
  oss << "%" + prefix + "_" + suffix << " = torch.prim.ListConstruct ";
  // %val_0, %val_1, ...
  interleave(
      ssaValueNames.begin(), ssaValueNames.end(),
      // each_fn:
      [&](const std::string &name) { oss << name; },
      // between_fn:
      [&] { oss << ", "; });
  oss << " : (";
  // !torch.int, !torch.int, ...
  interleave(
      ssaValueNames.begin(), ssaValueNames.end(),
      // each_fn:
      [&](const std::string &name) { oss << "!torch.int"; },
      // between_fn:
      [&] { oss << ", "; });
  oss << ") -> !torch.list<int>\n";

  return oss.str();
}

//===----------------------------------------------------------------------===//
//
// TensorAttr ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits a ranked tensor type in MLIR assembly representation.
//
// This expects ranked tensors (non-scalar) as we blanket generate a
// `!torch.vtensor` (or `!torch.tensor` if mutable) type. The caller
// is responsible to check for this. In the future we may want to extend
// this (or add new methods) for scalar types (such as `!torch.int` or
// `!torch.bool`).
//
// Example:
//
//    TensorAttr t;
//    t.setName("tensor")
//      .setDataType(DataType::Float)
//      .setDim({2, 3, 4})
//      .setStride({12, 1, 3})
//
//    t.getTensorTypeAsm(/*isValueTensor=*/true,
//                       /*useLogicalDims=*/true)
//        --> "!torch.vtensor<[2,3,4],f32>"
//
//    t.getTensorTypeAsm(/*isValueTensor=*/false,
//                       /*useLogicalDims=*/true)
//        --> "!torch.tensor<[2,3,4],f32>"
//
//    t.getTensorTypeAsm(/*isValueTensor=*/true,
//                       /*useLogicalDims=*/false)
//        --> "!torch.vtensor<[2,4,3],f32>"
//
//    t.getTensorTypeAsm(/*isValueTensor=*/false,
//                       /*useLogicalDims=*/false)
//        --> "!torch.tensor<[2,4,3],f32>"
inline std::string TensorAttr::getTensorTypeAsm(bool isValueTensor,
                                                bool useLogicalDims) const {
  assert(!isScalar() && "TensorAttr::getTensorTypeAsm expects a ranked tensor");
  assert(!getDim().empty() &&
         "TensorAttr::getTensorTypeAsm expects non-empty dims");
  assert(!getStride().empty() &&
         "TensorAttr::getTensorTypeAsm expects non-empty strides");
  assert(getDataType() != DataType::NotSet &&
         "TensorAttr::getTensorTypeAsm expects a valid data type");

  std::ostringstream oss;
  oss << (isValueTensor ? "!torch.vtensor<[" : "!torch.tensor<[");

  std::vector<int64_t> dims = useLogicalDims ? getDim() : getPhysicalDim();

  // Emit dims in logical or physical order.
  interleave(
      dims.begin(), dims.end(),
      // each_fn:
      [&](int64_t dim) { oss << dim; },
      // between_fn:
      [&] { oss << ","; });
  oss << "],";
  oss << kDataTypeToMlirTypeAsm.at(getDataType());
  oss << ">";
  return oss.str();
}

// Emits an MLIR SSA value name starting with the `%` sigil based off the
// TensorAttr name but only using alphanumeric / underscore [A-Za-z0-9_]
// characters.
//
// `foo_Bar::X0` becomes `%foo_BarX0` if `isOutputAliased=false`.
// `foo_Bar::X0` becomes `%foo_BarX0_` if `isOutputAliased=true`.
inline std::string TensorAttr::getValueNameAsm(bool isOutputAliased) const {
  assert(!getName().empty() &&
         "TensorAttr name must not be empty for `getValueNameAsm`");

  std::string filtered = getName();
  std::erase_if(filtered, // C++20
                [](unsigned char c) { return !(std::isalnum(c) || c == '_'); });
  return "%" + filtered + (isOutputAliased ? "_" : "");
}

//===----------------------------------------------------------------------===//
//
// Graph ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits Graph's operand names and types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      func.func @main(..., {}) -> ...
// with
//      "%arg0_image: !torch.vtensor<[16,128,64,64],f32>,
//       %arg1_filter: !torch.vtensor<[256,128,1,1],f32>"
//
// Order of operands is made to be deterministic, and it is
// determined by the sorting order used in `fullGraphInputsSorted_`
// which sorts based on the name on the TensorAttrs.
inline std::string Graph::getOperandNamesAndTypesAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphInputsSorted_.begin(), fullGraphInputsSorted_.end(),
      // each_fn:
      [&](const std::shared_ptr<TensorAttr> &input) {
        oss << input->getValueNameAsm() << ": " << input->getTensorTypeAsm();
      },
      // between_fn:
      [&] { oss << ", "; },
      // skip_fn:
      [&](const std::shared_ptr<TensorAttr> &input) {
        // We only use the tensor inputs and not scalar (constants) as those
        // wouldn't be part of the main func.func signature but embedded as
        // constants in the IR.
        return input->isScalar();
      });
  return oss.str();
}

// Emits Graph's result names and types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      func.func @main({}, ...) -> ...
// with
//      "%result: !torch.tensor<[16,256,64,64],f32>
//
// Order of results is made to be deterministic, and it is
// determined by the sorting order used in `fullGraphOutputsSorted_`
// which sorts based on the name on the TensorAttrs.
inline std::string Graph::getResultNamesAndTypesAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphOutputsSorted_.begin(), fullGraphOutputsSorted_.end(),
      // each_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        oss << output->getValueNameAsm(/*isOutputAliased=*/true) << ": "
            << output->getTensorTypeAsm(/*isValueTensor=*/false);
      },
      // between_fn:
      [&] { oss << ", "; },
      // skip_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        // We only want the final outputs in the return so ignore any virtual
        // tensors here as they're intermediates.
        return output->isVirtual();
      });
  return oss.str();
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the pre-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string Graph::emitNodePreAsm() const {
  constexpr std::string_view schema = R"(
module @module {{
  func.func @main({0}, {1}) attributes {{torch.assume_strict_symbolic_shapes}} {{
  )";

  const std::string output = std::format(schema,
                                         getResultNamesAndTypesAsm(), // {0}
                                         getOperandNamesAndTypesAsm() // {1}
  );

  return output;
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the post-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string Graph::emitNodePostAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphOutputsSorted_.begin(), fullGraphOutputsSorted_.end(),
      // each_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        oss << "torch.overwrite.tensor.contents "
            << output->getValueNameAsm(/*isOutputAliased=*/false)
            << " overwrites "
            << output->getValueNameAsm(/*isOutputAliased=*/true) << " : "
            << output->getTensorTypeAsm(/*isValueTensor=*/true) << ", "
            << output->getTensorTypeAsm(/*isValueTensor=*/false);
      },
      // between_fn:
      [&] { oss << "\n"; },
      // skip_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        // We only want the final outputs in the return so ignore any virtual
        // tensors here as they're intermediates.
        return output->isVirtual();
      });

  constexpr std::string_view schema = R"(
    {0}

    return
  }}
}}
  )";

  const std::string output = std::format(schema,
                                         oss.str() // {0}
  );

  return output;
}

//===----------------------------------------------------------------------===//
//
// ConvFPropNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits ConvFPropNode's operand names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution {}, ...
// with
//      "%arg0_image, %arg1_filter"
inline std::string ConvFPropNode::getOperandNamesAsm() const {
  return convFPropAttr.getX()->getValueNameAsm() + "_perm" + ", " +
         convFPropAttr.getW()->getValueNameAsm() + "_perm";
}

// Emits ConvFPropNode's operand types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution ... : {}, ...
// with
//      "!torch.vtensor<[16,128,64,64],f32>, !torch.vtensor<[256,128,1,1],f32>"
inline std::string ConvFPropNode::getOperandTypesAsm() const {
  return convFPropAttr.getX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true) +
         ", " +
         convFPropAttr.getW()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
}

// Emits ConvFPropNode's result names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      {} = torch.aten.convolution ...
// with
//      "%result"
inline std::string ConvFPropNode::getResultNamesAsm() const {
  return convFPropAttr.getY()->getValueNameAsm();
}

// Emits ConvFPropNode's result types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution ... -> {}
// with
//      "!torch.vtensor<[16,256,64,64],f32>"
inline std::string ConvFPropNode::getResultTypesAsm() const {
  return convFPropAttr.getY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
}

// Get groups in MLIR assembly format.
inline std::string ConvFPropNode::getGroupOpsAsm() const {
  constexpr size_t channelsIdx = 1;
  const int64_t inChannels = convFPropAttr.getX()->getDim()[channelsIdx];
  const int64_t filterChannels = convFPropAttr.getW()->getDim()[channelsIdx];
  int64_t groupCount = inChannels / filterChannels;

  return std::format("%groups_{} = torch.constant.int {}",
                     convFPropAttr.getName(), groupCount);
}

// Get strides in MLIR assembly format.
inline std::string ConvFPropNode::getStrideOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getStride(), /*prefix=*/"stride",
                            /*suffix=*/convFPropAttr.getName());
}

// Get padding in MLIR assembly format.
inline std::string ConvFPropNode::getPaddingOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getPadding(), /*prefix=*/"padding",
                            /*suffix=*/convFPropAttr.getName());
}

// Get dilation in MLIR assembly format.
inline std::string ConvFPropNode::getDilationOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getDilation(), /*prefix=*/"dilation",
                            /*suffix=*/convFPropAttr.getName());
}

// Get permute ops for input X in MLIR assembly format.
inline std::string ConvFPropNode::getPermuteXOpsAsm() const {
  std::ostringstream oss;

  const std::string prefix = "permute_X";
  const std::string suffix = convFPropAttr.getName();
  const std::shared_ptr<TensorAttr> xT = convFPropAttr.getX();

  // Emit permute dimensions based on layout.
  if (xT->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(xT->getDim().size()), prefix, suffix);
  else
    oss << getListOfIntOpsAsm(
        getChannelsLastToContiguousPermuteOrder(xT->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0}_perm = torch.aten.permute {0}, {1} : {2}, !torch.list<int> -> {3}
  )";

  const std::string output =
      std::format(schema,
                  xT->getValueNameAsm(),       // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  xT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/false), // {2}
                  xT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/true) // {3}
      );

  return oss.str() + output;
}

// Get permute ops for weight W in MLIR assembly format.
inline std::string ConvFPropNode::getPermuteWOpsAsm() const {
  std::ostringstream oss;

  const std::string prefix = "permute_W";
  const std::string suffix = convFPropAttr.getName();
  const std::shared_ptr<TensorAttr> wT = convFPropAttr.getW();

  // Emit permute dimensions based on layout.
  if (wT->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(wT->getDim().size()), prefix, suffix);
  else
    oss << getListOfIntOpsAsm(
        getChannelsLastToContiguousPermuteOrder(wT->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0}_perm = torch.aten.permute {0}, {1} : {2}, !torch.list<int> -> {3}
  )";

  const std::string output =
      std::format(schema,
                  wT->getValueNameAsm(),       // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  wT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/false), // {2}
                  wT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/true) // {3}
      );

  return oss.str() + output;
}

// Get permute ops for output Y in MLIR assembly format.
inline std::string ConvFPropNode::getPermuteYOpsAsm() const {
  std::ostringstream oss;

  const std::string prefix = "permute_Y";
  const std::string suffix = convFPropAttr.getName();
  const std::shared_ptr<TensorAttr> yT = convFPropAttr.getY();

  // Emit permute dimensions based on layout.
  if (yT->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(yT->getDim().size()), prefix, suffix);
  else
    oss << getListOfIntOpsAsm(
        getContiguousToChannelsLastPermuteOrder(yT->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0} = torch.aten.permute {0}_perm, {1} : {2}, !torch.list<int> -> {3}
  )";

  const std::string output =
      std::format(schema,
                  yT->getValueNameAsm(),       // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  yT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/true), // {2}
                  yT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/false) // {3}
      );

  return oss.str() + output;
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the pre-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string ConvFPropNode::emitNodePreAsm() const {
  // `torch.aten.convolution` signature from GeneratedTorchOps.td
  // https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td
  //
  //  def Torch_AtenConvolutionOp : Torch_Op<"aten.convolution", [
  //    ...
  //    let summary = "Generated op for `aten::convolution : (Tensor, Tensor,
  //    Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)`"; let
  //    arguments = (ins
  //      AnyTorchTensorType:$input,
  //      AnyTorchTensorType:$weight,
  //      AnyTorchOptionalTensorType:$bias,
  //      AnyTorchListOfTorchIntType:$stride,
  //      AnyTorchListOfTorchIntType:$padding,
  //      AnyTorchListOfTorchIntType:$dilation,
  //      Torch_BoolType:$transposed,
  //      AnyTorchListOfTorchIntType:$output_padding,
  //      Torch_IntType:$groups
  //    );
  //    let results = (outs
  //      AnyTorchOptionalTensorType:$result
  //    );
  //   ...
  constexpr std::string_view schema = R"(
    %bias_{0} = torch.constant.none
    %transposed_{0} = torch.constant.bool false
    %output_padding_{0} = torch.prim.ListConstruct  : () -> !torch.list<int>
    {1}
    {2}
    {3}
    {4}
    {5}
    {6}
    {7}_perm = torch.aten.convolution {8}, %bias_{0}, %stride_{0}, %padding_{0}, %dilation_{0}, %transposed_{0}, %output_padding_{0}, %groups_{0} : {9}, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> {10}
    {11}
    )";

  // Suffix the SSA names of internal values (constant attributes) using
  // the unique ConvFPropAttr name to avoid re-definition of names across
  // the overall MLIR assembly.
  std::string uniqueSSASuffix = convFPropAttr.getName();

  const std::string output = std::format(schema,
                                         uniqueSSASuffix,      // {0}
                                         getGroupOpsAsm(),     // {1}
                                         getStrideOpsAsm(),    // {2}
                                         getPaddingOpsAsm(),   // {3}
                                         getDilationOpsAsm(),  // {4}
                                         getPermuteXOpsAsm(),  // {5}
                                         getPermuteWOpsAsm(),  // {6}
                                         getResultNamesAsm(),  // {7}
                                         getOperandNamesAsm(), // {8}
                                         getOperandTypesAsm(), // {9}
                                         getResultTypesAsm(),  // {10}
                                         getPermuteYOpsAsm()   // {11}
  );

  return output;
}

//===----------------------------------------------------------------------===//
//
// ConvWGradNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits ConvWGradNode's operand names in MLIR assembly format.
// torch.aten.convolution_backward has a fixed op signature that takes 3 main
// args (dy, x, and w). The empty tensor (%empty_w_{suffix}) is required by
// torch.aten.convolution_backward for the w arg even when calculating weight
// gradient.
inline std::string ConvWGradNode::getOperandNamesAsm() const {
  const std::string suffix = convWGradAttr.getName();
  return convWGradAttr.getDY()->getValueNameAsm() + "_perm" + ", " +
         convWGradAttr.getX()->getValueNameAsm() + "_perm" + ", %empty_w_" +
         suffix;
}

// Emits ConvWGradNode's operand types in MLIR assembly format.
// Note: An operand for W is required by torch.aten.convolution_backward even
// when calculating weight gradient, so it's included after the DY and X types.
inline std::string ConvWGradNode::getOperandTypesAsm() const {
  return convWGradAttr.getDY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true) +
         ", " +
         convWGradAttr.getX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true) +
         ", " +
         convWGradAttr.getDW()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true);
}

// Emits ConvWGradNode's result names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      {} = torch.aten.convolution_backward ...
// with
//      "%result"
inline std::string ConvWGradNode::getResultNamesAsm() const {
  return convWGradAttr.getDW()->getValueNameAsm();
}

// Emits ConvWGradNode's result types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution_backward ... -> {}
// with
//      "!torch.vtensor<[256,128,1,1],f32>"
inline std::string ConvWGradNode::getResultTypesAsm() const {
  return convWGradAttr.getDW()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true);
}

// Get strides in MLIR assembly format.
inline std::string ConvWGradNode::getStrideOpsAsm() const {
  return getListOfIntOpsAsm(convWGradAttr.getStride(), /*prefix=*/"stride",
                            /*suffix=*/convWGradAttr.getName());
}

// Get padding in MLIR assembly format.
inline std::string ConvWGradNode::getPaddingOpsAsm() const {
  return getListOfIntOpsAsm(convWGradAttr.getPadding(), /*prefix=*/"padding",
                            /*suffix=*/convWGradAttr.getName());
}

// Get dilation in MLIR assembly format.
inline std::string ConvWGradNode::getDilationOpsAsm() const {
  return getListOfIntOpsAsm(convWGradAttr.getDilation(), /*prefix=*/"dilation",
                            /*suffix=*/convWGradAttr.getName());
}

// Get permute operations for DY tensor in MLIR assembly format.
inline std::string ConvWGradNode::getPermuteDYOpsAsm() const {
  std::ostringstream oss;
  const std::string prefix = "permute_DY";
  const std::string suffix = convWGradAttr.getName();
  const std::shared_ptr<TensorAttr> dyT = convWGradAttr.getDY();

  // Emit permute dimensions based on layout.
  if (dyT->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(dyT->getDim().size()), prefix,
        suffix);
  else
    oss << getListOfIntOpsAsm(
        getChannelsLastToContiguousPermuteOrder(dyT->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0}_perm = torch.aten.permute {0}, {1} : {2}, !torch.list<int> -> {3}
  )";

  const std::string output =
      std::format(schema,
                  dyT->getValueNameAsm(),      // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  dyT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/false), // {2}
                  dyT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/true) // {3}
      );

  return oss.str() + output;
}

// Get permute operations for X tensor in MLIR assembly format.
inline std::string ConvWGradNode::getPermuteXOpsAsm() const {
  std::ostringstream oss;
  const std::string prefix = "permute_X";
  const std::string suffix = convWGradAttr.getName();
  const std::shared_ptr<TensorAttr> xT = convWGradAttr.getX();

  // Emit permute dimensions based on layout.
  if (xT->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(xT->getDim().size()), prefix, suffix);
  else
    oss << getListOfIntOpsAsm(
        getChannelsLastToContiguousPermuteOrder(xT->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0}_perm = torch.aten.permute {0}, {1} : {2}, !torch.list<int> -> {3}
  )";

  const std::string output =
      std::format(schema,
                  xT->getValueNameAsm(),       // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  xT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/false), // {2}
                  xT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/true) // {3}
      );

  return oss.str() + output;
}

// Get permute operations for DW tensor in MLIR assembly format.
inline std::string ConvWGradNode::getPermuteDWOpsAsm() const {
  std::ostringstream oss;
  const std::string prefix = "permute_DW";
  const std::string suffix = convWGradAttr.getName();
  const std::shared_ptr<TensorAttr> dwT = convWGradAttr.getDW();

  // Emit permute dimensions based on layout.
  if (dwT->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(dwT->getDim().size()), prefix,
        suffix);
  else
    oss << getListOfIntOpsAsm(
        getContiguousToChannelsLastPermuteOrder(dwT->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0} = torch.aten.permute {0}_perm, {1} : {2}, !torch.list<int> -> {3}
  )";

  const std::string output =
      std::format(schema,
                  dwT->getValueNameAsm(),      // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  dwT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/true), // {2}
                  dwT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/false) // {3}
      );

  return oss.str() + output;
}

// `torch.aten.convolution_backward` requires an input for the weight even when
// calculating the gradient of the weight. Create an empty tensor with the same
// dimensions as the weight tensor.
inline std::string ConvWGradNode::getPermuteEmptyWOpsAsm() const {
  std::ostringstream oss;
  const std::string prefix = "empty_DW";
  const std::string suffix = convWGradAttr.getName();
  const std::shared_ptr<TensorAttr> dwT = convWGradAttr.getDW();

  oss << getListOfIntOpsAsm(dwT->getDim(), prefix, suffix);

  // Use `torch.aten.empty.memory_format` to create an empty tensor. It is the
  // simplest op to create a new tensor without having a pre-existing one
  // (then `torch.aten.empty_like` could be used).
  constexpr std::string_view schema = R"(
    %none_DW_{0} = torch.constant.none
    %dtype_DW_{0} = torch.constant.int {3}
    %empty_w_{0} = torch.aten.empty.memory_format {1}, %dtype_DW_{0}, %none_DW_{0}, %none_DW_{0}, %none_DW_{0}, %none_DW_{0} : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> {2}
  )";

  const torch_upstream::ScalarType dataType =
      kDataTypeToTorchType.at(dwT->getDataType());
  const std::string output =
      std::format(schema,
                  suffix,                      // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  dwT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/true), // {2}
                  std::to_string(static_cast<int>(dataType))      // {3}
      );

  return oss.str() + output;
}

inline std::string ConvWGradNode::emitNodePreAsm() const {
  constexpr std::string_view schema = R"(
    %bias_{0} = torch.constant.none
    %transposed_{0} = torch.constant.bool false
    %output_padding_{0} = torch.prim.ListConstruct  : () -> !torch.list<int>
    %groups_{0} = torch.constant.int 1
    {1}
    {2}
    {3}
    {4}
    {5}
    {6}
    %true_{0} = torch.constant.bool true
    %false_{0} = torch.constant.bool false
    %output_mask_{0} = torch.prim.ListConstruct %false_{0}, %true_{0}, %false_{0} : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
    %grad_input_{0}, {7}_perm, %grad_bias_{0} = torch.aten.convolution_backward {8}, %bias_{0}, %stride_{0}, %padding_{0}, %dilation_{0}, %transposed_{0}, %output_padding_{0}, %groups_{0}, %output_mask_{0} : {9}, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.none, {10}, !torch.none
    {11}
    )";

  // Suffix the SSA names of internal values (constant attributes) using
  // the unique ConvWGradAttr name to avoid re-definition of names across
  // the overall MLIR assembly.
  std::string uniqueSSASuffix = convWGradAttr.getName();

  const std::string output = std::format(schema,
                                         uniqueSSASuffix,          // {0}
                                         getStrideOpsAsm(),        // {1}
                                         getPaddingOpsAsm(),       // {2}
                                         getDilationOpsAsm(),      // {3}
                                         getPermuteDYOpsAsm(),     // {4}
                                         getPermuteXOpsAsm(),      // {5}
                                         getPermuteEmptyWOpsAsm(), // {6}
                                         getResultNamesAsm(),      // {7}
                                         getOperandNamesAsm(),     // {8}
                                         getOperandTypesAsm(),     // {9}
                                         getResultTypesAsm(),      // {10}
                                         getPermuteDWOpsAsm()      // {11}
  );

  return output;
}

//===----------------------------------------------------------------------===//
//
// ConvDGradNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits ConvDGradNode's operand names in MLIR assembly format.
// torch.aten.convolution_backward has a fixed op signature that takes 3 main
// args (dy, x, and w). The empty tensor (%empty_x_{suffix}) is required by
// torch.aten.convolution_backward for the x arg even when calculating data
// gradient, so it's included between DY and W operands.
inline std::string ConvDGradNode::getOperandNamesAsm() const {
  const std::string suffix = convDGradAttr.getName();
  return convDGradAttr.getDY()->getValueNameAsm() + "_perm" + ", %empty_x_" +
         suffix + ", " + convDGradAttr.getW()->getValueNameAsm() + "_perm";
}

// Emits ConvDGradNode's operand types in MLIR assembly format.
// Note: An operand for X is required by torch.aten.convolution_backward even
// when calculating data gradient, so it's included between DY and W operands.
inline std::string ConvDGradNode::getOperandTypesAsm() const {
  return convDGradAttr.getDY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true) +
         ", " +
         convDGradAttr.getDX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true) +
         ", " +
         convDGradAttr.getW()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
}

// Emits ConvDGradNode's result names in MLIR assembly format.
inline std::string ConvDGradNode::getResultNamesAsm() const {
  return convDGradAttr.getDX()->getValueNameAsm();
}

// Emits ConvDGradNode's result types in MLIR assembly format.
inline std::string ConvDGradNode::getResultTypesAsm() const {
  return convDGradAttr.getDX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                 /*useLogicalDims=*/true);
}

// Get strides in MLIR assembly format.
inline std::string ConvDGradNode::getStrideOpsAsm() const {
  return getListOfIntOpsAsm(convDGradAttr.getStride(), /*prefix=*/"stride",
                            /*suffix=*/convDGradAttr.getName());
}

// Get padding in MLIR assembly format.
inline std::string ConvDGradNode::getPaddingOpsAsm() const {
  return getListOfIntOpsAsm(convDGradAttr.getPadding(), /*prefix=*/"padding",
                            /*suffix=*/convDGradAttr.getName());
}

// Get dilation in MLIR assembly format.
inline std::string ConvDGradNode::getDilationOpsAsm() const {
  return getListOfIntOpsAsm(convDGradAttr.getDilation(), /*prefix=*/"dilation",
                            /*suffix=*/convDGradAttr.getName());
}

// Get permute operations for DY tensor in MLIR assembly format.
inline std::string ConvDGradNode::getPermuteDYOpsAsm() const {
  std::ostringstream oss;
  const std::string prefix = "permute_DY";
  const std::string suffix = convDGradAttr.getName();
  const std::shared_ptr<TensorAttr> dyT = convDGradAttr.getDY();

  // Emit permute dimensions based on layout.
  if (dyT->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(dyT->getDim().size()), prefix,
        suffix);
  else
    oss << getListOfIntOpsAsm(
        getChannelsLastToContiguousPermuteOrder(dyT->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0}_perm = torch.aten.permute {0}, {1} : {2}, !torch.list<int> -> {3}
  )";

  const std::string output =
      std::format(schema,
                  dyT->getValueNameAsm(),      // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  dyT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/false), // {2}
                  dyT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/true) // {3}
      );

  return oss.str() + output;
}

// Get permute operations for W tensor in MLIR assembly format.
inline std::string ConvDGradNode::getPermuteWOpsAsm() const {
  std::ostringstream oss;
  const std::string prefix = "permute_W";
  const std::string suffix = convDGradAttr.getName();
  const std::shared_ptr<TensorAttr> wT = convDGradAttr.getW();

  // Emit permute dimensions based on layout.
  if (wT->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(wT->getDim().size()), prefix, suffix);
  else
    oss << getListOfIntOpsAsm(
        getChannelsLastToContiguousPermuteOrder(wT->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0}_perm = torch.aten.permute {0}, {1} : {2}, !torch.list<int> -> {3}
  )";

  const std::string output =
      std::format(schema,
                  wT->getValueNameAsm(),       // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  wT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/false), // {2}
                  wT->getTensorTypeAsm(/*isValueTensor=*/true,
                                       /*useLogicalDims=*/true) // {3}
      );

  return oss.str() + output;
}

// Get permute operations for DX tensor in MLIR assembly format.
inline std::string ConvDGradNode::getPermuteDXOpsAsm() const {
  std::ostringstream oss;
  const std::string prefix = "permute_DX";
  const std::string suffix = convDGradAttr.getName();
  const std::shared_ptr<TensorAttr> dxT = convDGradAttr.getDX();

  // Emit permute dimensions based on layout.
  if (dxT->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(dxT->getDim().size()), prefix,
        suffix);
  else
    oss << getListOfIntOpsAsm(
        getContiguousToChannelsLastPermuteOrder(dxT->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0} = torch.aten.permute {0}_perm, {1} : {2}, !torch.list<int> -> {3}
  )";

  const std::string output =
      std::format(schema,
                  dxT->getValueNameAsm(),      // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  dxT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/true), // {2}
                  dxT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/false) // {3}
      );

  return oss.str() + output;
}

// `torch.aten.convolution_backward` requires an input for the image even when
// calculating the gradient of the image. Create an empty tensor with the same
// dimensions as DX.
inline std::string ConvDGradNode::getPermuteEmptyXOpsAsm() const {
  std::ostringstream oss;
  const std::string prefix = "empty_DX";
  const std::string suffix = convDGradAttr.getName();
  const std::shared_ptr<TensorAttr> dxT = convDGradAttr.getDX();

  oss << getListOfIntOpsAsm(dxT->getDim(), prefix, suffix);

  // Use `torch.aten.empty.memory_format` to create an empty tensor. It is the
  // simplest op to create a new tensor without having a pre-existing one
  // (then `torch.aten.empty_like` could be used).
  constexpr std::string_view schema = R"(
    %none_DX_{0} = torch.constant.none
    %dtype_DX_{0} = torch.constant.int {3}
    %empty_x_{0} = torch.aten.empty.memory_format {1}, %dtype_DX_{0}, %none_DX_{0}, %none_DX_{0}, %none_DX_{0}, %none_DX_{0} : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> {2}
  )";

  const torch_upstream::ScalarType dataType =
      kDataTypeToTorchType.at(dxT->getDataType());
  const std::string output =
      std::format(schema,
                  suffix,                      // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  dxT->getTensorTypeAsm(/*isValueTensor=*/true,
                                        /*useLogicalDims=*/true), // {2}
                  std::to_string(static_cast<int>(dataType))      // {3}
      );

  return oss.str() + output;
}

inline std::string ConvDGradNode::emitNodePreAsm() const {
  constexpr std::string_view schema = R"(
    %bias_{0} = torch.constant.none
    %transposed_{0} = torch.constant.bool false
    %output_padding_{0} = torch.prim.ListConstruct  : () -> !torch.list<int>
    %groups_{0} = torch.constant.int 1
    {1}
    {2}
    {3}
    {4}
    {5}
    {6}
    %true_{0} = torch.constant.bool true
    %false_{0} = torch.constant.bool false
    %output_mask_{0} = torch.prim.ListConstruct %true_{0}, %false_{0}, %false_{0} : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
    {7}_perm, %grad_weight_{0}, %grad_bias_{0} = torch.aten.convolution_backward {8}, %bias_{0}, %stride_{0}, %padding_{0}, %dilation_{0}, %transposed_{0}, %output_padding_{0}, %groups_{0}, %output_mask_{0} : {9}, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> {10}, !torch.none, !torch.none
    {11}
  )";

  // Suffix the SSA names of internal values (constant attributes) using
  // the unique ConvDGradAttr name to avoid re-definition of names across
  // the overall MLIR assembly.
  std::string uniqueSSASuffix = convDGradAttr.getName();

  const std::string output = std::format(schema,
                                         uniqueSSASuffix,          // {0}
                                         getStrideOpsAsm(),        // {1}
                                         getPaddingOpsAsm(),       // {2}
                                         getDilationOpsAsm(),      // {3}
                                         getPermuteDYOpsAsm(),     // {4}
                                         getPermuteWOpsAsm(),      // {5}
                                         getPermuteEmptyXOpsAsm(), // {6}
                                         getResultNamesAsm(),      // {7}
                                         getOperandNamesAsm(),     // {8}
                                         getOperandTypesAsm(),     // {9}
                                         getResultTypesAsm(),      // {10}
                                         getPermuteDXOpsAsm()      // {11}
  );
  return output;
}

//===----------------------------------------------------------------------===//
//
// PointwiseNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits PointwiseNode's operand names in MLIR assembly format.
inline std::string PointwiseNode::getOperandNamesAsm() const {
  std::ostringstream oss;
  const auto &in0 = pointwiseAttr.getIN_0();
  oss << in0->getValueNameAsm();
  if (const auto &in1 = pointwiseAttr.getIN_1())
    oss << ", " << in1->getValueNameAsm();
  if (const auto &in2 = pointwiseAttr.getIN_2())
    oss << ", " << in2->getValueNameAsm();
  return oss.str();
}

// Emits PointwiseNode's operand types in MLIR assembly format.
inline std::string PointwiseNode::getOperandTypesAsm() const {
  std::ostringstream oss;
  const auto &in0 = pointwiseAttr.getIN_0();
  oss << in0->getTensorTypeAsm();
  if (const auto &in1 = pointwiseAttr.getIN_1())
    oss << ", " << in1->getTensorTypeAsm();
  if (const auto &in2 = pointwiseAttr.getIN_2())
    oss << ", " << in2->getTensorTypeAsm();
  return oss.str();
}

// Emits PointwiseNode's result names in MLIR assembly format.
inline std::string PointwiseNode::getResultNamesAsm() const {
  return pointwiseAttr.getOUT_0()->getValueNameAsm();
}

// Emits PointwiseNode's result types in MLIR assembly format.
inline std::string PointwiseNode::getResultTypesAsm() const {
  return pointwiseAttr.getOUT_0()->getTensorTypeAsm();
}

// Emits PointwiseNode's result names and types in MLIR assembly format.
inline std::string PointwiseNode::getResultNamesAndTypesAsm() const {
  return getResultNamesAsm() + ": " + getResultTypesAsm();
}

inline std::string PointwiseNode::emitNodePreAsm() const {
  switch (pointwiseAttr.getMode()) {
  case PointwiseAttr::Mode::RELU_FWD: {
    constexpr std::string_view schema = R"(
    {0} = torch.aten.relu {1} : {2} -> {3}
    )";

    return std::format(schema,
                       getResultNamesAsm(),  // {0}
                       getOperandNamesAsm(), // {1}
                       getOperandTypesAsm(), // {2}
                       getResultTypesAsm()   // {3}
    );
  }
  case PointwiseAttr::Mode::ADD: {
    constexpr std::string_view schema = R"(
    %alpha_{0} = torch.constant.int 1
    {1} = torch.aten.add.Tensor {2}, %alpha_{0} : {3}, !torch.int -> {4}
    )";
    std::string uniqueSSASuffix = getName();

    return std::format(schema, uniqueSSASuffix, // {0}
                       getResultNamesAsm(),     // {1}
                       getOperandNamesAsm(),    // {2}
                       getOperandTypesAsm(),    // {3}
                       getResultTypesAsm()      // {4}
    );
  }
  case PointwiseAttr::Mode::DIV: {
    constexpr std::string_view schema = R"(
    {0} = torch.aten.div.Tensor {1} : {2} -> {3}
    )";
    return std::format(schema,
                       getResultNamesAsm(),  // {0}
                       getOperandNamesAsm(), // {1}
                       getOperandTypesAsm(), // {2}
                       getResultTypesAsm()   // {3}
    );
  }
  case PointwiseAttr::Mode::MUL: {
    constexpr std::string_view schema = R"(
    {0} = torch.aten.mul.Tensor {1} : {2} -> {3}
    )";
    return std::format(schema,
                       getResultNamesAsm(),  // {0}
                       getOperandNamesAsm(), // {1}
                       getOperandTypesAsm(), // {2}
                       getResultTypesAsm()   // {3}
    );
  }
  case PointwiseAttr::Mode::SUB: {
    constexpr std::string_view schema = R"(
    %alpha_{0} = torch.constant.int 1
    {1} = torch.aten.sub.Tensor {2}, %alpha_{0} : {3}, !torch.int -> {4}
    )";
    std::string uniqueSSASuffix = getName();

    return std::format(schema, uniqueSSASuffix, // {0}
                       getResultNamesAsm(),     // {1}
                       getOperandNamesAsm(),    // {2}
                       getOperandTypesAsm(),    // {3}
                       getResultTypesAsm()      // {4}
    );
  }
  default:
    assert(false && "Unsupported pointwise mode");
    return "";
  }
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_ASM_EMITTER_H
