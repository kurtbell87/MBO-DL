#pragma once

#include "analysis/analysis_result.hpp"
#include "analysis/multiple_comparison.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// SpearmanResult — result of a Spearman rank correlation computation
// ---------------------------------------------------------------------------
struct SpearmanResult {
    float correlation = 0.0f;
    float p_value = 1.0f;
    float ci_lower = 0.0f;
    float ci_upper = 0.0f;

    // For integration with Holm-Bonferroni
    float raw_p_value = 1.0f;
    float corrected_p_value = 1.0f;
};

// ---------------------------------------------------------------------------
// SpearmanAnalyzer — Spearman rank correlation with p-value and CI
// ---------------------------------------------------------------------------
class SpearmanAnalyzer {
public:
    SpearmanResult compute(const std::vector<float>& x,
                           const std::vector<float>& y) const {
        SpearmanResult result;
        size_t n = x.size();
        if (n != y.size() || n < 3) {
            result.correlation = 0.0f;
            result.p_value = 1.0f;
            result.ci_lower = 0.0f;
            result.ci_upper = 0.0f;
            return result;
        }

        auto rx = compute_ranks(x);
        auto ry = compute_ranks(y);

        // Check for zero variance in ranks (constant input)
        float rx_var = rank_variance(rx);
        float ry_var = rank_variance(ry);
        if (rx_var < 1e-10f || ry_var < 1e-10f) {
            result.correlation = 0.0f;
            result.p_value = 1.0f;
            result.ci_lower = 0.0f;
            result.ci_upper = 0.0f;
            return result;
        }

        // Pearson correlation of ranks
        result.correlation = pearson(rx, ry);
        result.correlation = std::max(-1.0f, std::min(1.0f, result.correlation));

        // P-value using t-distribution approximation
        float nf = static_cast<float>(n);
        float r = result.correlation;
        float one_minus_r2 = 1.0f - r * r;
        float t_stat;
        if (one_minus_r2 < 1e-10f) {
            // |r| ≈ 1 → t_stat → ∞ → p ≈ 0
            t_stat = 1e6f;
        } else {
            float denom = std::sqrt(one_minus_r2 / (nf - 2.0f));
            t_stat = (denom > 1e-10f) ? r / denom : 0.0f;
        }

        // Two-tailed p-value using normal approximation (for large n)
        float z = std::abs(t_stat);
        result.p_value = two_tailed_p(z);
        result.p_value = std::min(1.0f, result.p_value);
        result.raw_p_value = result.p_value;

        // 95% CI using Fisher z-transform
        float z_r = std::atanh(std::max(-0.999f, std::min(0.999f, r)));
        float se = 1.0f / std::sqrt(std::max(1.0f, nf - 3.0f));
        float z_crit = 1.96f;  // 95% CI
        result.ci_lower = std::tanh(z_r - z_crit * se);
        result.ci_upper = std::tanh(z_r + z_crit * se);

        return result;
    }

    // Analyze all feature × horizon × bar_type combinations with Holm-Bonferroni.
    std::vector<AnalysisResult> analyze_all(
        const std::vector<std::vector<float>>& features,
        const std::vector<std::vector<float>>& returns_per_horizon,
        const std::vector<std::string>& bar_types) const {

        std::vector<AnalysisResult> results;
        std::vector<float> raw_pvals;

        for (size_t bt = 0; bt < bar_types.size(); ++bt) {
            for (size_t h = 0; h < returns_per_horizon.size(); ++h) {
                for (size_t f = 0; f < features.size(); ++f) {
                    auto sr = compute(features[f], returns_per_horizon[h]);

                    AnalysisResult ar;
                    ar.point_estimate = sr.correlation;
                    ar.ci_lower = sr.ci_lower;
                    ar.ci_upper = sr.ci_upper;
                    ar.raw_p_value = sr.p_value;
                    ar.sample_count = static_cast<int>(features[f].size());
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
    static std::vector<float> compute_ranks(const std::vector<float>& data) {
        size_t n = data.size();
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return data[a] < data[b];
        });

        std::vector<float> ranks(n);
        size_t i = 0;
        while (i < n) {
            size_t j = i + 1;
            while (j < n && data[order[j]] == data[order[i]]) ++j;
            // Average rank for ties
            float avg_rank = 0.5f * (static_cast<float>(i + 1) + static_cast<float>(j));
            for (size_t k = i; k < j; ++k) {
                ranks[order[k]] = avg_rank;
            }
            i = j;
        }
        return ranks;
    }

    static float rank_variance(const std::vector<float>& ranks) {
        float sum = 0.0f;
        for (float r : ranks) sum += r;
        float mean = sum / static_cast<float>(ranks.size());
        float var = 0.0f;
        for (float r : ranks) {
            float d = r - mean;
            var += d * d;
        }
        return var;
    }

    static float pearson(const std::vector<float>& x, const std::vector<float>& y) {
        size_t n = x.size();
        float sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
        for (size_t i = 0; i < n; ++i) {
            sx += x[i];
            sy += y[i];
            sxy += x[i] * y[i];
            sxx += x[i] * x[i];
            syy += y[i] * y[i];
        }
        float nf = static_cast<float>(n);
        float cov = nf * sxy - sx * sy;
        float vx = nf * sxx - sx * sx;
        float vy = nf * syy - sy * sy;
        float denom = std::sqrt(vx * vy);
        if (denom < 1e-10f) return 0.0f;
        return cov / denom;
    }

    static float normal_cdf(float x) {
        return 0.5f * (1.0f + std::erf(static_cast<double>(x) / std::sqrt(2.0)));
    }

    // Two-tailed p-value using normal approximation, computed in double precision
    // to avoid underflow for large z values.
    static float two_tailed_p(float z_abs) {
        double z = static_cast<double>(z_abs);
        double p = std::erfc(z / std::sqrt(2.0));  // erfc = 2*(1-Phi(z)) for positive z
        p = std::max(p, 1e-30);  // ensure strictly positive
        return static_cast<float>(p);
    }
};
