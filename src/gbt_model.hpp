#pragma once

#include "gbt_features.hpp"
#include "trajectory_builder.hpp"

#include <xgboost/c_api.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// GBTOverfitResult — output of overfit_gbt()
// ---------------------------------------------------------------------------
struct GBTOverfitResult {
    float accuracy = 0.0f;
    int rounds = 0;
    bool success = false;
};

// ---------------------------------------------------------------------------
// GBTModel — XGBoost C API wrapper for multi-class classification
// ---------------------------------------------------------------------------
class GBTModel {
public:
    GBTModel() : booster_(nullptr) {}

    ~GBTModel() {
        if (booster_) {
            XGBoosterFree(booster_);
            booster_ = nullptr;
        }
    }

    // Non-copyable
    GBTModel(const GBTModel&) = delete;
    GBTModel& operator=(const GBTModel&) = delete;

    // Move semantics
    GBTModel(GBTModel&& other) noexcept : booster_(other.booster_) {
        other.booster_ = nullptr;
    }
    GBTModel& operator=(GBTModel&& other) noexcept {
        if (this != &other) {
            if (booster_) XGBoosterFree(booster_);
            booster_ = other.booster_;
            other.booster_ = nullptr;
        }
        return *this;
    }

    void train(const std::vector<std::array<float, GBT_FEATURE_DIM>>& features,
               const std::vector<int>& labels,
               int num_rounds = 100) {
        if (features.empty() || labels.empty()) {
            throw std::invalid_argument("Training data must not be empty");
        }
        if (features.size() != labels.size()) {
            throw std::invalid_argument("features.size() != labels.size()");
        }

        int n = static_cast<int>(features.size());

        // Convert labels to float
        std::vector<float> flabels(n);
        for (int i = 0; i < n; ++i) {
            flabels[i] = static_cast<float>(labels[i]);
        }

        // Create DMatrix (RAII — freed automatically)
        auto dmat = make_dmatrix(features);

        // Set labels
        check(XGDMatrixSetFloatInfo(dmat.handle, "label", flabels.data(), n));

        // Free old booster if any
        if (booster_) {
            XGBoosterFree(booster_);
            booster_ = nullptr;
        }

        // Create booster
        DMatrixHandle dmats[] = {dmat.handle};
        check(XGBoosterCreate(dmats, 1, &booster_));

        // Set parameters (from spec)
        check(XGBoosterSetParam(booster_, "objective", "multi:softmax"));
        check(XGBoosterSetParam(booster_, "num_class", "5"));
        check(XGBoosterSetParam(booster_, "max_depth", "10"));
        check(XGBoosterSetParam(booster_, "learning_rate", "0.1"));
        check(XGBoosterSetParam(booster_, "subsample", "1.0"));
        check(XGBoosterSetParam(booster_, "colsample_bytree", "1.0"));
        check(XGBoosterSetParam(booster_, "min_child_weight", "1"));
        check(XGBoosterSetParam(booster_, "seed", "42"));
        check(XGBoosterSetParam(booster_, "nthread", "1"));

        // Train
        for (int i = 0; i < num_rounds; ++i) {
            check(XGBoosterUpdateOneIter(booster_, i, dmat.handle));
        }
    }

    std::vector<int> predict(const std::vector<std::array<float, GBT_FEATURE_DIM>>& features) {
        if (!booster_) {
            throw std::runtime_error("Model not trained - call train() or load() first");
        }

        int n = static_cast<int>(features.size());

        // Create DMatrix (RAII — freed automatically)
        auto dmat = make_dmatrix(features);

        // Predict
        bst_ulong out_len = 0;
        const float* out_result = nullptr;
        check(XGBoosterPredict(booster_, dmat.handle, 0, 0, 0, &out_len, &out_result));

        std::vector<int> predictions(n);
        for (int i = 0; i < n; ++i) {
            predictions[i] = static_cast<int>(out_result[i]);
        }

        return predictions;
    }

    void save(const std::string& path) {
        if (!booster_) {
            throw std::runtime_error("No model to save");
        }
        int rc = XGBoosterSaveModel(booster_, path.c_str());
        if (rc != 0) {
            throw std::runtime_error("Failed to save model to: " + path);
        }
    }

    void load(const std::string& path) {
        if (booster_) {
            XGBoosterFree(booster_);
            booster_ = nullptr;
        }

        // Create an empty booster first
        check(XGBoosterCreate(nullptr, 0, &booster_));

        int rc = XGBoosterLoadModel(booster_, path.c_str());
        if (rc != 0) {
            XGBoosterFree(booster_);
            booster_ = nullptr;
            throw std::runtime_error("Failed to load model from: " + path);
        }
    }

private:
    BoosterHandle booster_;

    // RAII guard for DMatrixHandle — prevents leaks on exception
    struct DMatrixGuard {
        DMatrixHandle handle = nullptr;
        explicit DMatrixGuard(DMatrixHandle h) : handle(h) {}
        ~DMatrixGuard() { if (handle) XGDMatrixFree(handle); }
        DMatrixGuard(const DMatrixGuard&) = delete;
        DMatrixGuard& operator=(const DMatrixGuard&) = delete;
    };

    static DMatrixGuard make_dmatrix(
        const std::vector<std::array<float, GBT_FEATURE_DIM>>& features) {
        int n = static_cast<int>(features.size());
        std::vector<float> flat(n * GBT_FEATURE_DIM);
        for (int i = 0; i < n; ++i) {
            std::memcpy(&flat[i * GBT_FEATURE_DIM], features[i].data(),
                        GBT_FEATURE_DIM * sizeof(float));
        }
        DMatrixHandle dmat;
        check(XGDMatrixCreateFromMat(flat.data(), n, GBT_FEATURE_DIM, -1.0f, &dmat));
        return DMatrixGuard(dmat);
    }

    static void check(int rc) {
        if (rc != 0) {
            throw std::runtime_error(std::string("XGBoost error: ") + XGBGetLastError());
        }
    }
};

// ---------------------------------------------------------------------------
// overfit_gbt — train a GBT model on TrainingSamples to target accuracy
//
// Extracts GBT features from each TrainingSample's window using
// compute_gbt_features(), then trains a GBTModel.
// ---------------------------------------------------------------------------
inline GBTOverfitResult overfit_gbt(
    const std::vector<TrainingSample>& samples,
    int max_rounds = 1000,
    float target_accuracy = 0.99f)
{
    int n = static_cast<int>(samples.size());

    // Extract GBT features from each sample's encoded window.
    // Use the first GBT_FEATURE_DIM features from the last snapshot
    // in the window as the GBT feature vector.
    std::vector<std::array<float, GBT_FEATURE_DIM>> features(n);
    std::vector<int> labels(n);

    for (int i = 0; i < n; ++i) {
        labels[i] = samples[i].label;

        const auto& last_snap = samples[i].window[W - 1];
        for (int j = 0; j < GBT_FEATURE_DIM; ++j) {
            features[i][j] = last_snap[j];
        }
    }

    GBTOverfitResult result;
    GBTModel model;
    model.train(features, labels, max_rounds);

    auto predictions = model.predict(features);

    int correct = 0;
    for (int i = 0; i < n; ++i) {
        if (predictions[i] == labels[i]) correct++;
    }

    result.accuracy = static_cast<float>(correct) / static_cast<float>(n);
    result.rounds = max_rounds;
    result.success = result.accuracy >= target_accuracy;

    return result;
}
