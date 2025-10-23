// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Convolution wgrad; DY/X (NHWC), DW (KRSC); 2x2; padding and stride",
          "[conv][graph]") {
  int64_t n = 16, c = 128, h = 48, w = 32, k = 128, r = 2, s = 2;

  // Convolution parameters
  int64_t stride_h = 2, stride_w = 2;
  int64_t padding_h = 1, padding_w = 1;
  int64_t dilation_h = 1, dilation_w = 1;

  int64_t out_h =
      ((h + 2 * padding_h - dilation_h * (r - 1) - 1) / stride_h) + 1;
  int64_t out_w =
      ((w + 2 * padding_w - dilation_w * (s - 1) - 1) / stride_w) + 1;

  auto build_new_graph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_wgrad_sample_nhwc_krsc_2x2_pad_stride");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto DY =
        graph->tensor(TensorAttr()
                          .setName("dy")
                          .setDim({n, k, out_h, out_w})
                          .setStride({k * out_h * out_w, 1, k * out_w, k}));

    auto X = graph->tensor(TensorAttr()
                               .setName("x")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, 1, c * w, c}));

    auto wgradAttr = ConvWGradAttr()
                         .setStride({stride_h, stride_w})
                         .setPadding({padding_h, padding_w})
                         .setDilation({dilation_h, dilation_w})
                         .setName("conv_wgrad");

    auto DW = graph->convWGrad(DY, X, wgradAttr);
    DW->setName("dw")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({k, c, r, s});

    // Validate, infer missing properties
    REQUIRE(isOk(graph->validate()));

    // Compile
    REQUIRE(isOk(graph->compile(handle, /*remove=*/false)));

    return std::make_tuple(graph, DY, X, DW);
  };

  // Parameterize sample by backend and create device-specific handles.
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU)));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("amdgpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU)));
  }
#endif
  Handle &handle = *handlePtr;

  auto [graph, DY, X, DW] = build_new_graph(handle);

  // Allocate input buffers.
  // Use values of 1.0 for testing
  const float inputScalar = 1.0f;
  auto dyBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, DY, DataType::Float, inputScalar));
  auto xBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, X, DataType::Float, inputScalar));
  auto dwBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, DW, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {DY, dyBuf},
          {X, xBuf},
          {DW, dwBuf},
      };

  // Execute graph once.
  REQUIRE(isOk(graph->execute(variantPack)));

  // Read output buffer and validate values for 3x3, stride=2, padding=1.
  std::vector<float> dwVals;
  REQUIRE(isOk(dwBuf->read(handle, dwVals)));

  // TODO(IanWood1): give this an actual value. Need to handle padding.
  const float expected = 0.0f;
  for (auto val : dwVals)
    REQUIRE(val == expected);

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    REQUIRE(isOk(graph->execute(variantPack)));

  // Repeat output buffer checks.
  dwVals.clear();
  REQUIRE(isOk(dwBuf->read(handle, dwVals)));
  for (auto val : dwVals)
    REQUIRE(val == expected);
}
