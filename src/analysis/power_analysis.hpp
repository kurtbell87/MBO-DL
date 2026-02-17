#pragma once

#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// StratumInfo — describes a bar-type × horizon stratum
// ---------------------------------------------------------------------------
struct StratumInfo {
    std::string name;
    int sample_size = 0;
};

// ---------------------------------------------------------------------------
// PowerResult — per-stratum power analysis output
// ---------------------------------------------------------------------------
struct PowerResult {
    std::string name;
    int sample_size = 0;
    float detectable_r = 0.0f;
    float power_at_r005 = 0.0f;
};

// ---------------------------------------------------------------------------
// PowerAnalyzer — sample size and power calculations for Spearman correlation
//
// Based on the asymptotic approximation:
//   z = r * sqrt(n - 3) / sqrt(1 - r²)  (Fisher z-transform)
//   Power = P(Z > z_alpha - z_effect)
//
// For small r: n ≈ (z_alpha + z_beta)² / r² + 3
// ---------------------------------------------------------------------------
class PowerAnalyzer {
public:
    // Minimum sample size to detect Spearman r at given alpha and power.
    int min_sample_size_spearman(float effect_r, float alpha, float power) const {
        float z_alpha = z_from_alpha(alpha);
        float z_beta = z_from_power(power);

        // n ≈ ((z_alpha + z_beta) / arctanh(r))² + 3
        float w = std::atanh(std::abs(effect_r));
        if (w < 1e-10f) return 1000000;  // near-zero effect → huge n

        float n = std::pow((z_alpha + z_beta) / w, 2.0f) + 3.0f;
        return static_cast<int>(std::ceil(n));
    }

    // Detectable effect size given n, alpha, and power.
    float detectable_effect_size(int n, float alpha, float power) const {
        float z_alpha = z_from_alpha(alpha);
        float z_beta = z_from_power(power);

        // w = (z_alpha + z_beta) / sqrt(n - 3)
        float denom = std::sqrt(std::max(1.0f, static_cast<float>(n - 3)));
        float w = (z_alpha + z_beta) / denom;
        return std::tanh(w);
    }

    // Compute power given n, effect size r, and alpha.
    float compute_power(int n, float effect_r, float alpha) const {
        float z_alpha = z_from_alpha(alpha);
        float w = std::atanh(std::abs(effect_r));
        float se = 1.0f / std::sqrt(std::max(1.0f, static_cast<float>(n - 3)));
        float z_effect = w / se;
        float z = z_effect - z_alpha;
        return normal_cdf(z);
    }

    // Per-stratum power analysis.
    std::vector<PowerResult> per_stratum_power(const std::vector<StratumInfo>& strata,
                                                float alpha, float target_power) const {
        std::vector<PowerResult> results;
        results.reserve(strata.size());
        for (const auto& s : strata) {
            PowerResult pr;
            pr.name = s.name;
            pr.sample_size = s.sample_size;
            pr.detectable_r = detectable_effect_size(s.sample_size, alpha, target_power);
            pr.power_at_r005 = compute_power(s.sample_size, 0.05f, alpha);
            results.push_back(pr);
        }
        return results;
    }

private:
    // One-tailed z critical value from alpha (e.g., alpha=0.05 → z≈1.645)
    static float z_from_alpha(float alpha) {
        return -normal_quantile(alpha);
    }

    // z value from power (e.g., power=0.80 → z≈0.8416)
    static float z_from_power(float power) {
        return -normal_quantile(1.0f - power);
    }

    static float normal_cdf(float x) {
        return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
    }

    static float normal_quantile(float p) {
        if (p <= 0.0f) return -1e10f;
        if (p >= 1.0f) return 1e10f;
        float t;
        if (p < 0.5f) {
            t = std::sqrt(-2.0f * std::log(p));
        } else {
            t = std::sqrt(-2.0f * std::log(1.0f - p));
        }
        float c0 = 2.515517f, c1 = 0.802853f, c2 = 0.010328f;
        float d1 = 1.432788f, d2 = 0.189269f, d3 = 0.001308f;
        float result = t - (c0 + c1 * t + c2 * t * t) /
                           (1.0f + d1 * t + d2 * t * t + d3 * t * t * t);
        if (p < 0.5f) result = -result;
        return result;
    }
};
