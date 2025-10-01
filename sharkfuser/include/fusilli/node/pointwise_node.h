// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the pointwise nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_POINTWISE_NODE_H
#define FUSILLI_NODE_POINTWISE_NODE_H

#include "fusilli/attributes/pointwise_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/node.h"
#include "fusilli/support/logging.h"

#include <memory>
#include <string>

namespace fusilli {

class PointwiseNode : public NodeCRTP<PointwiseNode> {
public:
  PointwiseAttr pointwiseAttr;

  PointwiseNode(PointwiseAttr attr, const Context &ctx)
      : NodeCRTP(ctx), pointwiseAttr(std::move(attr)) {}

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const override final;
  std::string getOperandTypesAsm() const override final;
  std::string getResultNamesAsm() const override final;
  std::string getResultTypesAsm() const override final;
  std::string getResultNamesAndTypesAsm() const override final;

  const std::string &getName() const override final {
    return pointwiseAttr.getName();
  }
  Type getType() const override final { return Type::Pointwise; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating PointwiseNode '"
                           << pointwiseAttr.getName() << "'");
    FUSILLI_RETURN_ERROR_IF(
        pointwiseAttr.getMode() == PointwiseAttr::Mode::NOT_SET,
        ErrorCode::AttributeNotSet, "Pointwise mode not set");

    // Validate inputs based on mode
    switch (pointwiseAttr.getMode()) {
    case PointwiseAttr::Mode::RELU:
      FUSILLI_RETURN_ERROR_IF(!pointwiseAttr.getIN0(),
                              ErrorCode::AttributeNotSet,
                              "RELU mode requires IN0 input");
      break;
    case PointwiseAttr::Mode::BIAS:
      FUSILLI_RETURN_ERROR_IF(!pointwiseAttr.getIN0(),
                              ErrorCode::AttributeNotSet,
                              "BIAS mode requires IN0 input");
      FUSILLI_RETURN_ERROR_IF(!pointwiseAttr.getIN1(),
                              ErrorCode::AttributeNotSet,
                              "BIAS mode requires IN1 input");
      break;
    }

    // Validate output
    FUSILLI_RETURN_ERROR_IF(!pointwiseAttr.getOUT(), ErrorCode::AttributeNotSet,
                            "Pointwise operation requires output");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for PointwiseNode '"
                           << pointwiseAttr.getName() << "'");

    // Fill missing properties from context (including data types)
    pointwiseAttr.fillFromContext(context);

    // For pointwise operations, output properties are typically the same as
    // input
    switch (pointwiseAttr.getMode()) {
    case PointwiseAttr::Mode::RELU:
      // RELU preserves input shape, dtype, and strides
      if (pointwiseAttr.getIN0() && pointwiseAttr.getOUT()) {
        pointwiseAttr.getOUT()->setDim(pointwiseAttr.getIN0()->getDim());
        pointwiseAttr.getOUT()->setDataType(
            pointwiseAttr.getIN0()->getDataType());
        pointwiseAttr.getOUT()->setStride(pointwiseAttr.getIN0()->getStride());
      }
      break;
    case PointwiseAttr::Mode::BIAS:
      // BIAS preserves input shape, dtype, and strides (bias is broadcast)
      if (pointwiseAttr.getIN0() && pointwiseAttr.getOUT()) {
        pointwiseAttr.getOUT()->setDim(pointwiseAttr.getIN0()->getDim());
        pointwiseAttr.getOUT()->setDataType(
            pointwiseAttr.getIN0()->getDataType());
        pointwiseAttr.getOUT()->setStride(pointwiseAttr.getIN0()->getStride());
      }
      break;
    default:
      return ErrorObject(ErrorCode::InvalidAttribute, "Unknown pointwise mode");
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating PointwiseNode '"
                           << pointwiseAttr.getName() << "'");

    // Post-validation for pointwise operations
    switch (pointwiseAttr.getMode()) {
    case PointwiseAttr::Mode::RELU:
      // RELU: output should match input shape and dtype
      if (pointwiseAttr.getIN0() && pointwiseAttr.getOUT()) {
        FUSILLI_RETURN_ERROR_IF(pointwiseAttr.getIN0()->getDim() !=
                                    pointwiseAttr.getOUT()->getDim(),
                                ErrorCode::InvalidAttribute,
                                "RELU output dimensions must match input");
        FUSILLI_RETURN_ERROR_IF(pointwiseAttr.getIN0()->getDataType() !=
                                    pointwiseAttr.getOUT()->getDataType(),
                                ErrorCode::InvalidAttribute,
                                "RELU output dtype must match input");
      }
      break;
    case PointwiseAttr::Mode::BIAS:
      // BIAS: output should match input shape and dtype
      if (pointwiseAttr.getIN0() && pointwiseAttr.getOUT()) {
        FUSILLI_RETURN_ERROR_IF(pointwiseAttr.getIN0()->getDim() !=
                                    pointwiseAttr.getOUT()->getDim(),
                                ErrorCode::InvalidAttribute,
                                "BIAS output dimensions must match input");
        FUSILLI_RETURN_ERROR_IF(pointwiseAttr.getIN0()->getDataType() !=
                                    pointwiseAttr.getOUT()->getDataType(),
                                ErrorCode::InvalidAttribute,
                                "BIAS output dtype must match input");
      }
      break;
    default:
      return ErrorObject(ErrorCode::InvalidAttribute, "Unknown pointwise mode");
    }

    return ok();
  }
};
} // namespace fusilli

#endif // FUSILLI_NODE_POINTWISE_NODE_H
