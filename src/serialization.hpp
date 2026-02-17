#pragma once

// serialization.hpp — ONNX export and inference utilities for model serialization tests
//
// Provides:
//   export_to_onnx(model, sample_input, onnx_path)  — export to ONNX via Python subprocess
//   load_and_run_onnx(onnx_path, input_tensor)       — run inference via ONNX Runtime C++ API

#include <torch/torch.h>

#include <onnxruntime_cxx_api.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// export_to_onnx — export a torch::nn::Module to ONNX format
//
// Strategy:
//   1. Save all model parameters and buffers as a list of tensors
//   2. Save the sample input tensor
//   3. Write an inline Python script that:
//      a. Loads the tensors and builds an OrderedDict state_dict
//      b. Detects MLP vs CNN from state dict keys
//      c. Constructs the matching PyTorch model
//      d. Loads the state dict, traces, and exports to ONNX
//   4. Clean up temp files
// ---------------------------------------------------------------------------
template <typename ModelType>
void export_to_onnx(ModelType& model,
                    const torch::Tensor& sample_input,
                    const std::string& onnx_path) {
    model->eval();

    // Save state dict as tensor list + keys
    auto sd_path = onnx_path + ".sd.pt";
    auto input_path = onnx_path + ".input.pt";
    auto keys_path = onnx_path + ".keys.txt";
    auto script_path = onnx_path + ".export.py";

    std::vector<torch::Tensor> tensors;
    std::vector<std::string> keys;
    for (const auto& p : model->named_parameters(/*recurse=*/true)) {
        tensors.push_back(p.value().detach().cpu());
        keys.push_back(p.key());
    }
    for (const auto& b : model->named_buffers(/*recurse=*/true)) {
        tensors.push_back(b.value().detach().cpu());
        keys.push_back(b.key());
    }

    // Use pickle_save for Python-compatible format (torch::save uses JIT format)
    {
        c10::List<torch::Tensor> ivalue_list;
        for (auto& t : tensors) ivalue_list.push_back(t);
        auto bytes = torch::pickle_save(ivalue_list);
        std::ofstream of(sd_path, std::ios::binary);
        of.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    {
        auto bytes = torch::pickle_save(sample_input.detach().cpu());
        std::ofstream of(input_path, std::ios::binary);
        of.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    {
        std::ofstream kf(keys_path);
        for (const auto& k : keys) kf << k << "\n";
    }

    // Write Python export script
    {
        std::ofstream sf(script_path);
        sf << R"PY(
import torch, warnings, collections, sys
warnings.filterwarnings('ignore')

tensors = torch.load(sys.argv[1], weights_only=False)
with open(sys.argv[2]) as f:
    keys = [l.strip() for l in f if l.strip()]
sd = collections.OrderedDict(zip(keys, tensors))
x = torch.load(sys.argv[3], weights_only=False)
onnx_path = sys.argv[4]

# Detect model type from state dict keys
has_spatial = any('spatial' in k for k in keys)

if has_spatial:
    # CNN model — reconstruct architecture matching cnn_model.hpp
    class CNNModel(torch.nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.spatial_conv1 = torch.nn.Conv1d(2, 32, 3, padding=1)
            self.spatial_bn1 = torch.nn.BatchNorm1d(32)
            self.spatial_conv2 = torch.nn.Conv1d(32, 64, 3, padding=1)
            self.spatial_bn2 = torch.nn.BatchNorm1d(64)
            self.spatial_pool = torch.nn.AdaptiveAvgPool1d(1)
            self.temporal_conv1 = torch.nn.Conv1d(218, 128, 5, padding=2)
            self.temporal_bn1 = torch.nn.BatchNorm1d(128)
            self.temporal_conv2 = torch.nn.Conv1d(128, 256, 5, padding=2)
            self.temporal_bn2 = torch.nn.BatchNorm1d(256)
            self.temporal_conv3 = torch.nn.Conv1d(256, 256, 3, padding=1)
            self.temporal_bn3 = torch.nn.BatchNorm1d(256)
            self.temporal_pool = torch.nn.AdaptiveAvgPool1d(4)
            self.temporal_maxpool = torch.nn.AdaptiveMaxPool1d(4)
            self.pool_bn = torch.nn.BatchNorm1d(2048)
            self.pool_proj = torch.nn.Linear(2048, 256)
            self.classifier = torch.nn.Linear(256, num_classes)

        def forward(self, x):
            B = x.size(0)
            T = x.size(1)
            # Price ladder: bid[9..0] + ask[0..9]
            bid_prices = x[:, :, 0:10]
            bid_sizes = x[:, :, 10:20]
            ask_prices = x[:, :, 20:30]
            ask_sizes = x[:, :, 30:40]
            bid_prices_rev = bid_prices.flip(2)
            bid_sizes_rev = bid_sizes.flip(2)
            ladder_prices = torch.cat([bid_prices_rev, ask_prices], 2)
            ladder_sizes = torch.cat([bid_sizes_rev, ask_sizes], 2)
            ladder = torch.stack([ladder_prices, ladder_sizes], 3)
            spatial_in = ladder.reshape(B * T, 20, 2).permute(0, 2, 1)
            s = torch.relu(self.spatial_bn1(self.spatial_conv1(spatial_in)))
            s = torch.relu(self.spatial_bn2(self.spatial_conv2(s)))
            s = self.spatial_pool(s).squeeze(2).reshape(B, T, 64)
            # Trade features: indices 40..190 (150 features)
            trade = x[:, :, 40:190]
            # Scalar features: indices 190..194 (4 features)
            scalar = x[:, :, 190:194]
            combined = torch.cat([s, trade, scalar], 2)
            t = combined.permute(0, 2, 1)
            t = torch.relu(self.temporal_bn1(self.temporal_conv1(t)))
            t = torch.relu(self.temporal_bn2(self.temporal_conv2(t)))
            t = torch.relu(self.temporal_bn3(self.temporal_conv3(t)))
            avg_p = self.temporal_pool(t).flatten(1)
            max_p = self.temporal_maxpool(t).flatten(1)
            pooled = torch.cat([avg_p, max_p], 1)
            proj = torch.relu(self.pool_proj(self.pool_bn(pooled)))
            return self.classifier(proj)

    model = CNNModel()
else:
    # MLP model — reconstruct architecture matching mlp_model.hpp
    input_dim = sd['fc1.weight'].size(1)
    num_classes = sd['fc4.weight'].size(0)

    class MLPModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, 512)
            self.bn1 = torch.nn.BatchNorm1d(512)
            self.fc2 = torch.nn.Linear(512, 256)
            self.bn2 = torch.nn.BatchNorm1d(256)
            self.fc3 = torch.nn.Linear(256, 128)
            self.bn3 = torch.nn.BatchNorm1d(128)
            self.fc4 = torch.nn.Linear(128, num_classes)
            self.drop1 = torch.nn.Dropout(0.0)
            self.drop2 = torch.nn.Dropout(0.0)
            self.drop3 = torch.nn.Dropout(0.0)

        def forward(self, x):
            x = x.flatten(1)
            x = self.drop1(torch.relu(self.bn1(self.fc1(x))))
            x = self.drop2(torch.relu(self.bn2(self.fc2(x))))
            x = self.drop3(torch.relu(self.bn3(self.fc3(x))))
            return self.fc4(x)

    model = MLPModel()

model.load_state_dict(sd)
model.eval()
traced = torch.jit.trace(model, x)
torch.onnx.export(traced, x, onnx_path,
    opset_version=17,
    input_names=['input'],
    output_names=['output'])
)PY";
    }

    // Run the Python script
    auto log_path = onnx_path + ".log";
    std::string cmd = "python3 '" + script_path + "' '" + sd_path + "' '" +
                      keys_path + "' '" + input_path + "' '" + onnx_path +
                      "' 2>'" + log_path + "'";
    int rc = std::system(cmd.c_str());

    // Read error log before cleanup
    std::string error_log;
    if (rc != 0) {
        std::ifstream lf(log_path);
        if (lf.is_open()) {
            std::ostringstream ss;
            ss << lf.rdbuf();
            error_log = ss.str();
        }
    }

    // Clean up temp files
    std::filesystem::remove(sd_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(keys_path);
    std::filesystem::remove(script_path);
    std::filesystem::remove(log_path);

    if (rc != 0) {
        throw std::runtime_error("ONNX export failed: " + error_log);
    }
    if (!std::filesystem::exists(onnx_path)) {
        throw std::runtime_error("ONNX export produced no output file: " + onnx_path);
    }
}

// ---------------------------------------------------------------------------
// load_and_run_onnx — load an ONNX model and run inference via ONNX Runtime
//
// Returns output tensor matching the model's output shape.
// ---------------------------------------------------------------------------
inline torch::Tensor load_and_run_onnx(const std::string& onnx_path,
                                       const torch::Tensor& input_tensor) {
    // Ensure input is contiguous float32 on CPU
    auto input = input_tensor.contiguous().to(torch::kFloat32).to(torch::kCPU);

    // Set up ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "serialization_test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, onnx_path.c_str(), session_options);

    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);

    // Build input shape
    auto sizes = input.sizes();
    std::vector<int64_t> input_shape(sizes.begin(), sizes.end());

    // Create ORT tensor from the torch tensor data
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info,
        input.data_ptr<float>(),
        static_cast<size_t>(input.numel()),
        input_shape.data(),
        input_shape.size());

    // Run inference
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_ort, 1,
        output_names, 1);

    // Convert output to torch tensor (clone to own the data)
    auto& output_ort = outputs[0];
    auto type_info = output_ort.GetTensorTypeAndShapeInfo();
    auto output_shape = type_info.GetShape();
    auto output_data = output_ort.GetTensorData<float>();

    std::vector<int64_t> torch_shape(output_shape.begin(), output_shape.end());
    auto result = torch::from_blob(
        const_cast<float*>(output_data),
        torch_shape,
        torch::kFloat32).clone();

    return result;
}
