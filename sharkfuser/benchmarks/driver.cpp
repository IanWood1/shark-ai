// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <CLI/CLI.hpp>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// For CLI11 Option Validators
const auto NonNegativeInteger =
    CLI::Range(int64_t{0}, std::numeric_limits<int64_t>::max());
const auto PositiveInteger =
    CLI::Range(int64_t{1}, std::numeric_limits<int64_t>::max());
const auto ValidConvLayout = CLI::IsMember({"NCHW", "NHWC", "NCDHW", "NDHWC"});

enum class ConvMode {
  FWD = 1,
  DGRAD = 2,
  WGRAD = 4,
};

struct ConvBenchmarkParams {
  // Input tensor dimensions
  int64_t batch_size;
  int64_t in_channels;
  int64_t in_depth;
  int64_t in_height;
  int64_t in_width;

  // Filter dimensions
  int64_t out_channels;
  int64_t fil_depth;
  int64_t fil_height;
  int64_t fil_width;

  // Convolution parameters
  int64_t stride_depth;
  int64_t stride_height;
  int64_t stride_width;
  int64_t pad_depth;
  int64_t pad_height;
  int64_t pad_width;
  int64_t dilation_depth;
  int64_t dilation_height;
  int64_t dilation_width;

  // Layout and configuration
  std::string input_layout;
  std::string output_layout;
  std::string filter_layout;
  int64_t spatial_dims;
  bool use_bias;
  int64_t iterations;
  DataType io_type;

  // Methods to calculate dimensions and strides
  std::vector<int64_t> getInputDims() const {
    return (spatial_dims == 2)
               ? std::vector<int64_t>{batch_size, in_channels, in_height,
                                      in_width}
               : std::vector<int64_t>{batch_size, in_channels, in_depth,
                                      in_height, in_width};
  }

  std::vector<int64_t> getFilterDims() const {
    return (spatial_dims == 2)
               ? std::vector<int64_t>{out_channels, in_channels, fil_height,
                                      fil_width}
               : std::vector<int64_t>{out_channels, in_channels, fil_depth,
                                      fil_height, fil_width};
  }

  std::vector<int64_t> getOutputDims() const {
    if (spatial_dims == 2) {
      int64_t out_h = (in_height + 2 * pad_height -
                       dilation_height * (fil_height - 1) - 1) /
                          stride_height +
                      1;
      int64_t out_w =
          (in_width + 2 * pad_width - dilation_width * (fil_width - 1) - 1) /
              stride_width +
          1;
      return std::vector<int64_t>{batch_size, out_channels, out_h, out_w};
    } else {
      int64_t out_d =
          (in_depth + 2 * pad_depth - dilation_depth * (fil_depth - 1) - 1) /
              stride_depth +
          1;
      int64_t out_h = (in_height + 2 * pad_height -
                       dilation_height * (fil_height - 1) - 1) /
                          stride_height +
                      1;
      int64_t out_w =
          (in_width + 2 * pad_width - dilation_width * (fil_width - 1) - 1) /
              stride_width +
          1;
      return std::vector<int64_t>{batch_size, out_channels, out_d, out_h,
                                  out_w};
    }
  }

  std::vector<int64_t> getInputStride() const {
    if (spatial_dims == 2) {
      return (input_layout == "NCHW")
                 ? std::vector<int64_t>{in_channels * in_height * in_width,
                                        in_height * in_width, in_width, 1}
                 : std::vector<int64_t>{in_channels * in_height * in_width, 1,
                                        in_channels * in_width, in_channels};
    } else {
      return (input_layout == "NCDHW")
                 ? std::vector<int64_t>{in_channels * in_depth * in_height *
                                            in_width,
                                        in_depth * in_height * in_width,
                                        in_height * in_width, in_width, 1}
                 : std::vector<int64_t>{in_channels * in_depth * in_height *
                                            in_width,
                                        1, in_channels * in_height * in_width,
                                        in_width * in_channels, in_channels};
    }
  }

  std::vector<int64_t> getFilterStride() const {
    if (spatial_dims == 2) {
      return (filter_layout == "NCHW")
                 ? std::vector<int64_t>{in_channels * fil_height * fil_width,
                                        fil_height * fil_width, fil_width, 1}
                 : std::vector<int64_t>{in_channels * fil_height * fil_width, 1,
                                        fil_width * in_channels, in_channels};
    } else {
      return (filter_layout == "NCDHW")
                 ? std::vector<int64_t>{in_channels * fil_depth * fil_height *
                                            fil_width,
                                        fil_depth * fil_height * fil_width,
                                        fil_height * fil_width, fil_width, 1}
                 : std::vector<int64_t>{in_channels * fil_depth * fil_height *
                                            fil_width,
                                        1, fil_height * fil_width * in_channels,
                                        fil_width * in_channels, in_channels};
    }
  }

  std::vector<int64_t> getOutputStride() const {
    auto outDims = getOutputDims();
    if (spatial_dims == 2) {
      return (output_layout == "NCHW")
                 ? std::vector<int64_t>{out_channels * outDims[2] * outDims[3],
                                        outDims[2] * outDims[3], outDims[3], 1}
                 : std::vector<int64_t>{out_channels * outDims[2] * outDims[3],
                                        1, out_channels * outDims[3],
                                        out_channels};
    } else {
      return (output_layout == "NCDHW")
                 ? std::vector<int64_t>{out_channels * outDims[2] * outDims[3] *
                                            outDims[4],
                                        outDims[2] * outDims[3] * outDims[4],
                                        outDims[3] * outDims[4], outDims[4], 1}
                 : std::vector<int64_t>{
                       out_channels * outDims[2] * outDims[3] * outDims[4], 1,
                       out_channels * outDims[3] * outDims[4],
                       out_channels * outDims[4], out_channels};
    }
  }

  std::vector<int64_t> getConvStride() const {
    return (spatial_dims == 2)
               ? std::vector<int64_t>{stride_height, stride_width}
               : std::vector<int64_t>{stride_depth, stride_height,
                                      stride_width};
  }

  std::vector<int64_t> getConvPadding() const {
    return (spatial_dims == 2)
               ? std::vector<int64_t>{pad_height, pad_width}
               : std::vector<int64_t>{pad_depth, pad_height, pad_width};
  }

  std::vector<int64_t> getConvDilation() const {
    return (spatial_dims == 2)
               ? std::vector<int64_t>{dilation_height, dilation_width}
               : std::vector<int64_t>{dilation_depth, dilation_height,
                                      dilation_width};
  }

  std::vector<int64_t> getBiasDims() const {
    return (spatial_dims == 2) ? std::vector<int64_t>{1, out_channels, 1, 1}
                               : std::vector<int64_t>{1, out_channels, 1, 1, 1};
  }

  std::vector<int64_t> getBiasStride() const {
    if (spatial_dims == 2) {
      return (input_layout == "NCHW")
                 ? std::vector<int64_t>{out_channels, 1, 1, 1}
                 : std::vector<int64_t>{out_channels, 1, out_channels,
                                        out_channels};
    } else {
      return (input_layout == "NCDHW")
                 ? std::vector<int64_t>{out_channels, 1, 1, 1, 1}
                 : std::vector<int64_t>{out_channels, 1, out_channels,
                                        out_channels, out_channels};
    }
  }

  std::string getGraphName(const std::string &name) const {
    return std::format("{}_n{}_c{}_d{}_h{}_w{}_k{}_z{}_y{}_x{"
                       "}_t{}_u{}_v{}_o{}"
                       "_p{}_q{}_m{}_l{}_j{}_S{}_I{}_O{}_F{}_bias{}",
                       name, batch_size, in_channels, in_depth, in_height,
                       in_width, out_channels, fil_depth, fil_height, fil_width,
                       stride_depth, stride_height, stride_width, pad_depth,
                       pad_height, pad_width, dilation_depth, dilation_height,
                       dilation_width, spatial_dims, input_layout,
                       output_layout, filter_layout, use_bias);
  }
};

ErrorObject benchmark_conv_fprop(const ConvBenchmarkParams &params) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif

  // Build attributes based on 2D/3D conv and layouts using struct methods.
  auto xDims = params.getInputDims();
  auto wDims = params.getFilterDims();
  auto xStride = params.getInputStride();
  auto wStride = params.getFilterStride();
  auto convStride = params.getConvStride();
  auto convPadding = params.getConvPadding();
  auto convDilation = params.getConvDilation();
  auto biasDims = params.getBiasDims();
  auto biasStride = params.getBiasStride();

  // Build graph for the given handle (device), validate and compile it.
  auto graph = std::make_shared<Graph>();

  // Set unique name to prevent concurrent invocations of the benchmark driver
  // from polluting the same cache files leading to race conditions.
  graph->setName(params.getGraphName("benchmark_conv_fprop"));

  // Types on the graph are kept at fp32 but we explicitly set
  // individual tensor types below based on configuration. These
  // types hence don't matter much and are used only to infer
  // missing type annotations on tensors.
  graph->setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto X = graph->tensor(TensorAttr()
                             .setName("input")
                             .setDim(xDims)
                             .setStride(xStride)
                             .setDataType(params.io_type));

  auto W = graph->tensor(TensorAttr()
                             .setName("filter")
                             .setDim(wDims)
                             .setStride(wStride)
                             .setDataType(params.io_type));

  auto conv_attr = ConvFPropAttr()
                       .setStride(convStride)
                       .setPadding(convPadding)
                       .setDilation(convDilation)
                       .setName("conv_fprop");

  auto Y = graph->convFProp(X, W, conv_attr);
  Y->setDataType(params.io_type);

  std::shared_ptr<TensorAttr> B;
  if (params.use_bias) {
    B = graph->tensor(TensorAttr()
                          .setName("bias")
                          .setDim(biasDims)
                          .setStride(biasStride)
                          .setDataType(params.io_type));
    auto biasAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
    Y = graph->pointwise(Y, B, biasAttr);
    Y->setDataType(params.io_type);
  }
  Y->setOutput(true).setDataType(params.io_type);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph->validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));

  // Allocate input, weight and output buffers.
  auto xBuf =
      FUSILLI_TRY(allocateBufferOfType(handle, X, params.io_type, 1.0f));
  auto wBuf =
      FUSILLI_TRY(allocateBufferOfType(handle, W, params.io_type, 1.0f));
  auto yBuf =
      FUSILLI_TRY(allocateBufferOfType(handle, Y, params.io_type, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {Y, yBuf},
      };

  if (params.use_bias) {
    auto bBuf =
        FUSILLI_TRY(allocateBufferOfType(handle, B, params.io_type, 1.0f));
    variantPack.insert({B, bBuf});
  }

  // Execute graph a few times.
  for (size_t i = 0; i < params.iterations; i++)
    FUSILLI_CHECK_ERROR(graph->execute(variantPack));

  return ok();
}

ErrorObject benchmark_conv_wgrad(const ConvBenchmarkParams &params) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif

  // Build attributes based on 2D/3D conv and layouts using struct methods.
  // For WGrad: DY has output dimensions, X has input dimensions, DW has filter
  // dimensions
  auto dyDims =
      params.getOutputDims(); // DY has same dims as output of forward pass
  auto xDims =
      params.getInputDims(); // X has same dims as input of forward pass
  auto dwDims =
      params.getFilterDims(); // DW has same dims as filter of forward pass

  auto dyStride = params.getOutputStride(); // DY stride based on output layout
  auto xStride = params.getInputStride();   // X stride based on input layout
  auto dwStride = params.getFilterStride(); // DW stride based on filter layout

  auto convStride = params.getConvStride();
  auto convPadding = params.getConvPadding();
  auto convDilation = params.getConvDilation();

  // Build graph for the given handle (device), validate and compile it.
  auto graph = std::make_shared<Graph>();

  // Set unique name to prevent concurrent invocations of the benchmark driver
  // from polluting the same cache files leading to race conditions.
  auto graphName = params.getGraphName("benchmark_conv_wgrad");
  graph->setName(graphName);

  // Types on the graph are kept at fp32 but we explicitly set
  // individual tensor types below based on configuration. These
  // types hence don't matter much and are used only to infer
  // missing type annotations on tensors.
  graph->setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  // DY tensor (gradient tensor) - has same dimensions as output of forward pass
  auto DY = graph->tensor(
      TensorAttr().setName("dy").setDim(dyDims).setStride(dyStride).setDataType(
          params.io_type));

  // X tensor (input tensor) - has same dimensions as input of forward pass
  auto X = graph->tensor(
      TensorAttr().setName("x").setDim(xDims).setStride(xStride).setDataType(
          params.io_type));

  auto conv_attr = ConvWGradAttr()
                       .setStride(convStride)
                       .setPadding(convPadding)
                       .setDilation(convDilation)
                       .setName("conv_wgrad");

  auto DW = graph->convWGrad(DY, X, conv_attr);
  DW->setDataType(params.io_type)
      .setOutput(true)
      .setDim(dwDims)
      .setStride(dwStride);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph->validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));

  // Allocate input and output buffers.
  auto dyBuf =
      FUSILLI_TRY(allocateBufferOfType(handle, DY, params.io_type, 1.0f));
  auto xBuf =
      FUSILLI_TRY(allocateBufferOfType(handle, X, params.io_type, 1.0f));
  auto dwBuf =
      FUSILLI_TRY(allocateBufferOfType(handle, DW, params.io_type, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {DY, dyBuf},
          {X, xBuf},
          {DW, dwBuf},
      };

  // Execute graph a few times.
  for (size_t i = 0; i < params.iterations; i++)
    FUSILLI_CHECK_ERROR(graph->execute(variantPack));

  return ok();
}

int main(int argc, char **argv) {
  CLI::App mainApp{"Fusilli Benchmark Driver"};
  mainApp.require_subcommand(1);

  int64_t iter;
  mainApp.add_option("--iter,-i", iter, "Benchmark iterations")
      ->required()
      ->check(PositiveInteger);

  // Conv flags are kept in sync with MIOpen's ConvDriver:
  // https://github.com/ROCm/rocm-libraries/blob/db0544fb61f2c7bd5a86dce98d4963420c1c741a/projects/miopen/driver/conv_driver.hpp#L878
  CLI::App *convApp =
      mainApp.add_subcommand("conv", "Fusilli Benchmark Forward Convolution");

  // CLI Options:
  int64_t n, c, d, h, w, k, z, y, x, t, u, v, o, p, q, m, l, j, S;
  std::string I, F, O;
  ConvMode mode;
  convApp->add_option("--batchsize,-n", n, "Input batch size")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_channels,-c", c, "Input channels")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_d", d, "Input depth")
      ->default_val("-1")
      ->check(PositiveInteger);
  convApp->add_option("--in_h,-H", h, "Input height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_w,-W", w, "Input width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--out_channels,-k", k, "Output channels")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--fil_d", z, "Filter depth")
      ->default_val("-1")
      ->check(PositiveInteger);
  convApp->add_option("--fil_h,-y", y, "Filter height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--fil_w,-x", x, "Filter width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--conv_stride_d", t, "Conv stride depth")
      ->default_val("-1")
      ->check(PositiveInteger);
  convApp->add_option("--conv_stride_h,-u", u, "Conv stride height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--conv_stride_w,-v", v, "Conv stride width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--pad_d", o, "Conv padding depth")
      ->default_val("-1")
      ->check(NonNegativeInteger);
  convApp->add_option("--pad_h,-p", p, "Conv padding height")
      ->required()
      ->check(NonNegativeInteger);
  convApp->add_option("--pad_w,-q", q, "Conv padding width")
      ->required()
      ->check(NonNegativeInteger);
  convApp->add_option("--dilation_d", m, "Conv dilation depth")
      ->default_val("-1")
      ->check(PositiveInteger);
  convApp->add_option("--dilation_h,-l", l, "Conv dilation height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--dilation_w,-j", j, "Conv dilation width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_layout", I, "Input layout")
      ->required()
      ->check(ValidConvLayout);
  convApp->add_option("--fil_layout", F, "Filter layout")
      ->required()
      ->check(ValidConvLayout);
  convApp->add_option("--out_layout", O, "Output layout")
      ->required()
      ->check(ValidConvLayout);
  convApp
      ->add_option("--spatial_dim", S,
                   "Number of spatial dimensions (2 for conv2d, 3 for conv3d)")
      ->required()
      ->check(CLI::IsMember({2, 3}));
  convApp
      ->add_option("--conv_mode", mode,
                   "Conv mode (1 = FWD, 2 = DGRAD, 4 = WGRAD)")
      ->default_val(ConvMode::FWD)
      ->check(CLI::IsMember({ConvMode::FWD, ConvMode::WGRAD, ConvMode::DGRAD}));

  // CLI Flags:
  bool fp16{false}, bf16{false}, bias{false};
  auto f1 = convApp->add_flag("--fp16", fp16, "Run fp16 convolution");
  auto f2 = convApp->add_flag("--bf16", bf16, "Run bf16 convolution");
  // Can't specify both flags.
  f1->excludes(f2);
  convApp->add_flag("--bias,-b", bias, "Run with bias");

  CLI11_PARSE(mainApp, argc, argv);

  // Additional validation of convApp options (apart from default CLI checks)
  if (S == 2) {
    // Reject 3D layouts for 2D conv
    if (I.size() != 4 || F.size() != 4 || O.size() != 4) {
      std::cerr << "Detected at least one invalid {input, filter, output} "
                   "layout for 2D convolution."
                << std::endl;
      return 1;
    }
  }
  if (S == 3) {
    // Reject 2D layouts for 3D conv
    if (I.size() != 5 || F.size() != 5 || O.size() != 5) {
      std::cerr << "Detected at least one invalid {input, filter, output} "
                   "layout for 3D convolution."
                << std::endl;
      return 1;
    }
    // Reject default (sentinel) values for optional args in 3D conv
    if (d == -1 || z == -1 || t == -1 || o == -1 || m == -1) {
      std::cerr << "Detected at least one of {in_d, fil_d, conv_stride_d, "
                   "pad_d, dilation_d} that was not set for 3D convolution."
                << std::endl;
      return 1;
    }
  }

  std::cout << "Fusilli Benchmark started..." << std::endl;

  if (convApp->parsed()) {
    DataType convIOType;
    if (fp16)
      convIOType = DataType::Half;
    else if (bf16)
      convIOType = DataType::BFloat16;
    else
      // When unspecified, default to fp32 conv.
      convIOType = DataType::Float;

    ConvBenchmarkParams params{.batch_size = n,
                               .in_channels = c,
                               .in_depth = d,
                               .in_height = h,
                               .in_width = w,
                               .out_channels = k,
                               .fil_depth = z,
                               .fil_height = y,
                               .fil_width = x,
                               .stride_depth = t,
                               .stride_height = u,
                               .stride_width = v,
                               .pad_depth = o,
                               .pad_height = p,
                               .pad_width = q,
                               .dilation_depth = m,
                               .dilation_height = l,
                               .dilation_width = j,
                               .input_layout = I,
                               .output_layout = O,
                               .filter_layout = F,
                               .spatial_dims = S,
                               .use_bias = bias,
                               .iterations = iter,
                               .io_type = convIOType};

    ErrorObject status;
    switch (mode) {
    case ConvMode::FWD:
      status = benchmark_conv_fprop(params);
      break;
    case ConvMode::WGRAD:
      status = benchmark_conv_wgrad(params);
      break;
    case ConvMode::DGRAD:
      std::cerr << "DGRAD mode not yet implemented" << std::endl;
      return 1;
    default:
      std::cerr << "Unknown conv mode: " << static_cast<int>(mode) << std::endl;
      return 1;
    }

    if (isError(status)) {
      std::cerr << "Fusilli Benchmark failed: " << status << std::endl;
      return 1;
    }
  }

  std::cout << "Fusilli Benchmark complete!" << std::endl;
  return 0;
}
