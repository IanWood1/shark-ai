// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// pointwise nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace fusilli {

class PointwiseAttr : public AttributesCRTP<PointwiseAttr> {
public:
  // Names for Tensor Inputs and Outputs (doesn't include constant attributes).
  enum class InputNames { IN0, IN1 };
  enum class OutputNames { OUT };

  enum class Mode { RELU, BIAS, NOT_SET };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN0)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN1)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(PointwiseAttr, OutputNames, OUT)

  PointwiseAttr &setMode(Mode mode) {
    mode_ = mode;
    return *this;
  }

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN0)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN1)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, OUT)

  Mode getMode() const { return mode_; }

private:
  Mode mode_ = Mode::NOT_SET;
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H
