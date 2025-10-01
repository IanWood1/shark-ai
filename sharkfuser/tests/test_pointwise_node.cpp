// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

using namespace fusilli;

TEST_CASE("PointwiseNode getName correctly propagates the attribute name",
          "[pointwise_node]") {
  Context ctx;
  PointwiseAttr attr;
  attr.setName("foo_pointwise");

  PointwiseNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_pointwise");
}

TEST_CASE("PointwiseNode getType returns correct type", "[pointwise_node]") {
  Context ctx;
  PointwiseAttr attr;
  attr.setMode(PointwiseAttr::Mode::RELU);

  PointwiseNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::Pointwise);
}

TEST_CASE("PointwiseNode preValidateNode detects missing mode",
          "[pointwise_node]") {
  Context ctx;

  SECTION("Mode not set") {
    PointwiseAttr attr;
    PointwiseNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Pointwise mode not set");
  }

  SECTION("Mode set to RELU without inputs") {
    PointwiseAttr attr;
    attr.setMode(PointwiseAttr::Mode::RELU);
    PointwiseNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "RELU mode requires IN0 input");
  }

  SECTION("Mode set to BIAS without second input") {
    PointwiseAttr attr;
    attr.setMode(PointwiseAttr::Mode::BIAS);
    auto in0 = std::make_shared<TensorAttr>(1.0f);
    attr.setIN0(in0);
    PointwiseNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "BIAS mode requires IN1 input");
  }
}

TEST_CASE("PointwiseNode inferPropertiesNode works correctly",
          "[pointwise_node]") {
  Context ctx;
  PointwiseAttr attr;
  attr.setMode(PointwiseAttr::Mode::RELU);

  PointwiseNode node(std::move(attr), ctx);

  auto status = node.inferPropertiesNode();
  REQUIRE(!isError(status));
}

TEST_CASE("PointwiseNode postValidateNode works correctly",
          "[pointwise_node]") {
  Context ctx;
  PointwiseAttr attr;
  attr.setMode(PointwiseAttr::Mode::RELU);

  PointwiseNode node(std::move(attr), ctx);

  auto status = node.postValidateNode();
  REQUIRE(!isError(status));
}

TEST_CASE("PointwiseNode with tensor attributes", "[pointwise_node]") {
  Context ctx;
  PointwiseAttr attr;

  attr.setMode(PointwiseAttr::Mode::RELU);

  auto in0 = std::make_shared<TensorAttr>(1.0f);
  auto in1 = std::make_shared<TensorAttr>(2.0f);
  auto out = std::make_shared<TensorAttr>(3.0f);

  attr.setIN0(in0).setIN1(in1).setOUT(out);

  PointwiseNode node(std::move(attr), ctx);

  // Verify the node has access to the attributes
  REQUIRE(node.pointwiseAttr.getIN0() == in0);
  REQUIRE(node.pointwiseAttr.getIN1() == in1);
  REQUIRE(node.pointwiseAttr.getOUT() == out);
  REQUIRE(node.pointwiseAttr.getMode() == PointwiseAttr::Mode::RELU);

  // Verify tensor properties
  REQUIRE(node.pointwiseAttr.getIN0()->getDataType() == DataType::Float);
  REQUIRE(node.pointwiseAttr.getIN1()->getDataType() == DataType::Float);
  REQUIRE(node.pointwiseAttr.getOUT()->getDataType() == DataType::Float);

  REQUIRE(node.pointwiseAttr.getIN0()->getDim() == std::vector<int64_t>{1});
  REQUIRE(node.pointwiseAttr.getIN1()->getDim() == std::vector<int64_t>{1});
  REQUIRE(node.pointwiseAttr.getOUT()->getDim() == std::vector<int64_t>{1});
}
