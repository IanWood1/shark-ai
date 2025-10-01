// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

using namespace fusilli;

TEST_CASE("PointwiseAttr default constructor", "[pointwise_attr]") {
  PointwiseAttr attr;
  REQUIRE(attr.getMode() == PointwiseAttr::Mode::NOT_SET);
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("PointwiseAttr setters and getters", "[pointwise_attr]") {
  PointwiseAttr attr;
  PointwiseAttr::Mode mode = PointwiseAttr::Mode::RELU;

  attr.setMode(mode);

  REQUIRE(attr.getMode() == mode);

  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());

  auto in0 = std::make_shared<TensorAttr>(1.0f);
  auto in1 = std::make_shared<TensorAttr>(2.0f);
  auto out = std::make_shared<TensorAttr>(3.0f);

  attr.setIN0(in0).setIN1(in1).setOUT(out);

  REQUIRE(attr.inputs.size() == 2);
  REQUIRE(attr.outputs.size() == 1);

  REQUIRE(attr.getIN0() == in0);
  REQUIRE(attr.getIN1() == in1);
  REQUIRE(attr.getOUT() == out);

  REQUIRE(attr.getIN0()->getDataType() == DataType::Float);
  REQUIRE(attr.getIN1()->getDataType() == DataType::Float);
  REQUIRE(attr.getOUT()->getDataType() == DataType::Float);

  REQUIRE(attr.getIN0()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getIN1()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getOUT()->getDim() == std::vector<int64_t>{1});

  REQUIRE(attr.getIN0()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getIN1()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getOUT()->getStride() == std::vector<int64_t>{1});

  REQUIRE(attr.getIN0()->isScalar() == true);
  REQUIRE(attr.getIN1()->isScalar() == true);
  REQUIRE(attr.getOUT()->isScalar() == true);

  REQUIRE(attr.getIN0()->isVirtual() == false);
  REQUIRE(attr.getIN1()->isVirtual() == false);
  REQUIRE(attr.getOUT()->isVirtual() == false);
}

TEST_CASE("PointwiseAttr mode validation", "[pointwise_attr]") {
  PointwiseAttr attr;

  SECTION("Default mode is NOT_SET") {
    REQUIRE(attr.getMode() == PointwiseAttr::Mode::NOT_SET);
  }

  SECTION("Set mode to RELU") {
    attr.setMode(PointwiseAttr::Mode::RELU);
    REQUIRE(attr.getMode() == PointwiseAttr::Mode::RELU);
  }

  SECTION("Set mode to BIAS") {
    attr.setMode(PointwiseAttr::Mode::BIAS);
    REQUIRE(attr.getMode() == PointwiseAttr::Mode::BIAS);
  }

  SECTION("Chaining setMode") {
    attr.setMode(PointwiseAttr::Mode::RELU);
    REQUIRE(attr.getMode() == PointwiseAttr::Mode::RELU);
    
    // Verify chaining works
    auto &result = attr.setMode(PointwiseAttr::Mode::RELU);
    REQUIRE(&result == &attr);
  }
}
