#pragma once

#include "analysis/analysis_result.hpp"
#include "analysis/multiple_comparison.hpp"
#include "features/bar_features.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// quantile_discretize — assign each value to a quantile bin (0..num_bins-1)
// ---------------------------------------------------------------------------
inline std::vector<int> quantile_discretize(const std::vector<float>& values, int num_bins) {
    int n = static_cast<int>(values.size());
    if (n == 0) return {};

    // Create sorted index
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return values[a] < values[b];
    });

    std::vector<int> bins(n, 0);
    int effective_bins = std::min(num_bins, n);

    // Check for constant values
    if (values[order[0]] == values[order[n - 1]]) {
        // All values identical → all in bin 0
        return bins;
    }

    // Assign bins proportionally
    for (int i = 0; i < n; ++i) {
        int bin = static_cast<int>(static_cast<float>(i) * effective_bins / n);
        bin = std::min(bin, effective_bins - 1);
        bins[order[i]] = bin;
    }

    return bins;
}

// ---------------------------------------------------------------------------
// MIResult — mutual information computation result
// ---------------------------------------------------------------------------
struct MIResult {
    float mi_bits = 0.0f;
    float null_95th_percentile = 0.0f;
    float excess_mi = 0.0f;
    int null_shuffle_count = 0;
    int sample_count = 0;

    // For AnalysisResult integration
    float raw_p_value = 1.0f;
    float corrected_p_value = 1.0f;
};

// ---------------------------------------------------------------------------
// MIAnalyzer — mutual information computation with bootstrap null
// ---------------------------------------------------------------------------
class MIAnalyzer {
public:
    MIAnalyzer() : rng_(std::random_device{}()) {}
    explicit MIAnalyzer(uint32_t seed) : rng_(seed) {}

    // Compute raw MI between discretized feature and labels.
    MIResult compute_mi(const std::vector<float>& feature,
                        const std::vector<float>& labels,
                        int num_bins) {
        MIResult result;
        result.sample_count = static_cast<int>(feature.size());
        auto bins = quantile_discretize(feature, num_bins);
        result.mi_bits = mi_from_bins(bins, labels);
        return result;
    }

    // Compute MI with bootstrapped null distribution.
    MIResult compute_mi_with_null(const std::vector<float>& feature,
                                   const std::vector<float>& labels,
                                   int num_bins,
                                   int n_shuffles = 1000) {
        MIResult result;
        result.sample_count = static_cast<int>(feature.size());
        auto bins = quantile_discretize(feature, num_bins);
        result.mi_bits = mi_from_bins(bins, labels);
        result.null_shuffle_count = n_shuffles;

        // Bootstrap null: shuffle labels and recompute MI
        std::vector<float> shuffled_labels = labels;
        std::vector<float> null_mis;
        null_mis.reserve(n_shuffles);

        for (int s = 0; s < n_shuffles; ++s) {
            std::shuffle(shuffled_labels.begin(), shuffled_labels.end(), rng_);
            null_mis.push_back(mi_from_bins(bins, shuffled_labels));
        }

        // 95th percentile
        std::sort(null_mis.begin(), null_mis.end());
        int idx_95 = static_cast<int>(0.95f * n_shuffles);
        idx_95 = std::min(idx_95, n_shuffles - 1);
        result.null_95th_percentile = null_mis[idx_95];
        result.excess_mi = result.mi_bits - result.null_95th_percentile;

        // p-value: fraction of null >= observed
        int count_ge = 0;
        for (float nm : null_mis) {
            if (nm >= result.mi_bits) count_ge++;
        }
        result.raw_p_value = static_cast<float>(count_ge + 1) /
                             static_cast<float>(n_shuffles + 1);

        return result;
    }

    // Compute MI from BarFeatureRows, excluding warmup rows.
    MIResult compute_from_rows(const std::vector<BarFeatureRow>& rows,
                                const std::string& feature_name,
                                const std::string& return_name,
                                int num_bins) {
        std::vector<float> features, returns;

        for (const auto& row : rows) {
            if (row.is_warmup) continue;
            float f = row.get_feature_value(feature_name);
            float r = row.get_return_value(return_name);
            if (std::isnan(f) || std::isnan(r)) continue;
            features.push_back(f);
            // Convert return to sign for MI
            returns.push_back(r > 0 ? 1.0f : 0.0f);
        }

        MIResult result;
        result.sample_count = static_cast<int>(features.size());
        if (features.empty()) {
            result.mi_bits = std::numeric_limits<float>::quiet_NaN();
            return result;
        }

        auto bins = quantile_discretize(features, num_bins);
        result.mi_bits = mi_from_bins(bins, returns);
        return result;
    }

    // Analyze all feature × horizon × bar_type combinations.
    std::vector<AnalysisResult> analyze_all(
        const std::vector<std::vector<float>>& features,
        const std::vector<std::vector<float>>& returns_per_horizon,
        const std::vector<std::string>& bar_types) {

        std::vector<AnalysisResult> results;
        std::vector<float> raw_pvals;

        for (size_t bt = 0; bt < bar_types.size(); ++bt) {
            for (size_t h = 0; h < returns_per_horizon.size(); ++h) {
                for (size_t f = 0; f < features.size(); ++f) {
                    auto mi_result = compute_mi_with_null(
                        features[f], returns_per_horizon[h], 5, 1000);

                    AnalysisResult ar;
                    ar.point_estimate = mi_result.mi_bits;
                    ar.raw_p_value = mi_result.raw_p_value;
                    ar.sample_count = mi_result.sample_count;
                    results.push_back(ar);
                    raw_pvals.push_back(ar.raw_p_value);
                }
            }
        }

        // Apply Holm-Bonferroni
        auto corrected = holm_bonferroni_correct(raw_pvals);
        for (size_t i = 0; i < results.size(); ++i) {
            results[i].corrected_p_value = corrected[i];
            results[i].survives_correction = (corrected[i] < 0.05f);
            results[i].is_suggestive = (results[i].raw_p_value < 0.05f &&
                                        !results[i].survives_correction);
        }

        return results;
    }

private:
    std::mt19937 rng_;


    // Compute MI in bits from bin assignments and labels.
    static float mi_from_bins(const std::vector<int>& bins,
                               const std::vector<float>& labels) {
        int n = static_cast<int>(bins.size());
        if (n == 0) return 0.0f;

        // Count joint frequencies
        // First determine unique bins and labels
        int max_bin = *std::max_element(bins.begin(), bins.end());
        int n_bins = max_bin + 1;

        // Assume binary labels (0, 1) for return sign
        int n_labels = 2;

        // Joint distribution
        std::vector<int> joint(n_bins * n_labels, 0);
        std::vector<int> bin_counts(n_bins, 0);
        std::vector<int> label_counts(n_labels, 0);

        for (int i = 0; i < n; ++i) {
            int b = bins[i];
            int l = static_cast<int>(labels[i]);
            if (l < 0) l = 0;
            if (l >= n_labels) l = n_labels - 1;
            joint[b * n_labels + l]++;
            bin_counts[b]++;
            label_counts[l]++;
        }

        // MI = sum p(x,y) * log2(p(x,y) / (p(x)*p(y)))
        float nf = static_cast<float>(n);
        float mi = 0.0f;
        for (int b = 0; b < n_bins; ++b) {
            for (int l = 0; l < n_labels; ++l) {
                if (joint[b * n_labels + l] == 0) continue;
                float pxy = static_cast<float>(joint[b * n_labels + l]) / nf;
                float px = static_cast<float>(bin_counts[b]) / nf;
                float py = static_cast<float>(label_counts[l]) / nf;
                mi += pxy * std::log2(pxy / (px * py));
            }
        }

        return std::max(0.0f, mi);
    }
};
