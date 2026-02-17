#pragma once

#include "features/bar_features.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// QuintileInfo — info about one quintile bucket
// ---------------------------------------------------------------------------
struct QuintileInfo {
    int quintile_index = 0;
    int count = 0;
    float mean_feature_value = 0.0f;
};

// ---------------------------------------------------------------------------
// ConditionalReturnResult — output of conditional return analysis
// ---------------------------------------------------------------------------
struct ConditionalReturnResult {
    std::vector<float> quintile_means;  // Mean return per quintile (5 values)
    float q5_minus_q1 = 0.0f;          // Monotonicity measure
    float t_statistic = 0.0f;          // T-stat for Q5 vs Q1
    float t_p_value = 1.0f;
    int sample_count = 0;
};

// ---------------------------------------------------------------------------
// ConditionalReturnAnalyzer
// ---------------------------------------------------------------------------
class ConditionalReturnAnalyzer {
public:
    // Compute quintile buckets from feature values.
    std::vector<QuintileInfo> compute_quintiles(const std::vector<float>& feature) const {
        int n = static_cast<int>(feature.size());

        // Sort indices by feature value
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return feature[a] < feature[b];
        });

        std::vector<QuintileInfo> quintiles(5);
        for (int q = 0; q < 5; ++q) {
            quintiles[q].quintile_index = q;
        }

        // Assign each observation to a quintile
        std::vector<int> assignments(n);
        for (int i = 0; i < n; ++i) {
            int q = std::min(4, static_cast<int>(static_cast<float>(i) * 5.0f / n));
            assignments[order[i]] = q;
            quintiles[q].count++;
        }

        // Compute mean feature value per quintile
        std::vector<float> sums(5, 0.0f);
        for (int i = 0; i < n; ++i) {
            sums[assignments[i]] += feature[i];
        }
        for (int q = 0; q < 5; ++q) {
            if (quintiles[q].count > 0) {
                quintiles[q].mean_feature_value = sums[q] / static_cast<float>(quintiles[q].count);
            }
        }

        return quintiles;
    }

    // Analyze feature-return relationship.
    ConditionalReturnResult analyze(const std::vector<float>& feature,
                                    const std::vector<float>& returns) const {
        ConditionalReturnResult result;
        int n = static_cast<int>(feature.size());
        result.sample_count = n;

        if (n < 5) return result;

        // Sort indices by feature value
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return feature[a] < feature[b];
        });

        // Assign to quintiles
        std::vector<int> assignments(n);
        for (int i = 0; i < n; ++i) {
            int q = std::min(4, static_cast<int>(static_cast<float>(i) * 5.0f / n));
            assignments[order[i]] = q;
        }

        // Compute mean return per quintile
        std::vector<float> return_sums(5, 0.0f);
        std::vector<int> counts(5, 0);
        for (int i = 0; i < n; ++i) {
            return_sums[assignments[i]] += returns[i];
            counts[assignments[i]]++;
        }

        result.quintile_means.resize(5);
        for (int q = 0; q < 5; ++q) {
            result.quintile_means[q] = (counts[q] > 0)
                ? return_sums[q] / static_cast<float>(counts[q])
                : 0.0f;
        }

        // Monotonicity
        result.q5_minus_q1 = result.quintile_means[4] - result.quintile_means[0];

        // T-test: Q5 vs Q1
        std::vector<float> q1_returns, q5_returns;
        for (int i = 0; i < n; ++i) {
            if (assignments[i] == 0) q1_returns.push_back(returns[i]);
            if (assignments[i] == 4) q5_returns.push_back(returns[i]);
        }

        result.t_statistic = welch_t_stat(q5_returns, q1_returns);
        result.t_p_value = t_p_value(result.t_statistic,
            static_cast<int>(q1_returns.size() + q5_returns.size() - 2));

        return result;
    }

    // Analyze from BarFeatureRows, excluding warmup.
    ConditionalReturnResult analyze_from_rows(
        const std::vector<BarFeatureRow>& rows,
        const std::string& feature_name,
        const std::string& return_name) const {

        std::vector<float> features, returns;
        for (const auto& row : rows) {
            if (row.is_warmup) continue;
            features.push_back(row.get_feature_value(feature_name));
            returns.push_back(row.get_return_value(return_name));
        }

        auto result = analyze(features, returns);
        result.sample_count = static_cast<int>(features.size());
        return result;
    }

private:
    static float welch_t_stat(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty()) return 0.0f;

        float mean_a = 0.0f, mean_b = 0.0f;
        for (float v : a) mean_a += v;
        for (float v : b) mean_b += v;
        mean_a /= static_cast<float>(a.size());
        mean_b /= static_cast<float>(b.size());

        float var_a = 0.0f, var_b = 0.0f;
        for (float v : a) { float d = v - mean_a; var_a += d * d; }
        for (float v : b) { float d = v - mean_b; var_b += d * d; }
        float na = static_cast<float>(a.size());
        float nb = static_cast<float>(b.size());
        var_a /= (na > 1.0f ? na - 1.0f : 1.0f);
        var_b /= (nb > 1.0f ? nb - 1.0f : 1.0f);

        float se = std::sqrt(var_a / na + var_b / nb);
        if (se < 1e-12f) return 0.0f;
        return (mean_a - mean_b) / se;
    }

    static float t_p_value(float t_stat, int df) {
        if (df <= 0) return 1.0f;
        // Two-tailed using normal approximation, double precision to avoid underflow
        double z = std::abs(static_cast<double>(t_stat));
        double p = std::erfc(z / std::sqrt(2.0));
        p = std::max(p, 1e-30);
        return static_cast<float>(std::min(p, 1.0));
    }
};
