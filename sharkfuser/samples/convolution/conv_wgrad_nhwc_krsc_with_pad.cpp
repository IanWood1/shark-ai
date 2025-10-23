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

TEST_CASE("Convolution wgrad; DY/X (NHWC), DW (KRSC); 3x3; same padding",
          "[conv][graph]") {
  int64_t n = 16, c = 128, h = 64, w = 32, k = 256, r = 3, s = 3;

  auto build_new_graph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_wgrad_sample_nhwc_krsc_3x3_samepad");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto DY = graph->tensor(TensorAttr()
                                .setName("dy")
                                .setDim({n, k, h, w})
                                .setStride({k * h * w, 1, k * w, k})); // NHWC

    auto X = graph->tensor(TensorAttr()
                               .setName("x")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, 1, c * w, c})); // NHWC

    auto wgradAttr = ConvWGradAttr()
                         .setStride({1, 1})
                         .setPadding({1, 1})
                         .setDilation({1, 1})
                         .setName("conv_wgrad");

    auto DW = graph->convWGrad(DY, X, wgradAttr);
    DW->setName("dw")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({k, c, r, s});

    // Validate, infer missing properties
    REQUIRE(isOk(graph->validate()));

    // Compile
    REQUIRE(isOk(graph->compile(handle, /*remove=*/true)));

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

  // Read output buffer and validate values for 3x3, stride=1, same padding.
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
