#pragma once

#include "features/bar_features.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// DecayType — classification of signal decay pattern
// ---------------------------------------------------------------------------
enum class DecayType {
    SHORT_HORIZON_SIGNAL,
    REGIME_INDICATOR,
    NO_SIGNAL,
};

// ---------------------------------------------------------------------------
// DecayCurve — correlation at multiple horizons
// ---------------------------------------------------------------------------
struct DecayCurve {
    std::vector<int> horizons;
    std::vector<float> correlations;
    int sample_count = 0;
};

// ---------------------------------------------------------------------------
// DecayAnalyzer — computes decay curves and classifies signal persistence
// ---------------------------------------------------------------------------
class DecayAnalyzer {
public:
    // Standard horizons per spec
    static const std::vector<int>& standard_horizons() {
        static const std::vector<int> h = {1, 2, 5, 10, 20, 50, 100};
        return h;
    }

    // Compute decay curve for a single feature.
    DecayCurve compute_decay(const std::vector<BarFeatureRow>& rows,
                              const std::string& feature_name) const {
        // Filter non-warmup rows
        std::vector<float> feature_vals;
        std::vector<float> returns_1, returns_5, returns_20, returns_100;

        for (const auto& row : rows) {
            if (row.is_warmup) continue;
            feature_vals.push_back(row.get_feature_value(feature_name));
            returns_1.push_back(row.fwd_return_1);
            returns_5.push_back(row.fwd_return_5);
            returns_20.push_back(row.fwd_return_20);
            returns_100.push_back(row.fwd_return_100);
        }

        int n = static_cast<int>(feature_vals.size());

        DecayCurve curve;
        curve.horizons = standard_horizons();
        curve.sample_count = n;

        if (n < 10) {
            curve.correlations.assign(7, 0.0f);
            return curve;
        }

        // Map each target horizon to the closest available return vector.
        // Available: fwd_return_{1, 5, 20, 100}. Target: {1, 2, 5, 10, 20, 50, 100}.
        const std::vector<float>* available_returns[] = {
            &returns_1, &returns_5, &returns_20, &returns_100
        };
        constexpr int available_horizons[] = {1, 5, 20, 100};
        constexpr int n_available = 4;

        for (int h : curve.horizons) {
            // Find closest available horizon
            int best = 0;
            int best_dist = std::abs(h - available_horizons[0]);
            for (int k = 1; k < n_available; ++k) {
                int dist = std::abs(h - available_horizons[k]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = k;
                }
            }
            float corr = pearson_corr(feature_vals, *available_returns[best]);
            curve.correlations.push_back(corr);
        }

        return curve;
    }

    // Classify decay pattern.
    DecayType classify_decay(const DecayCurve& curve) const {
        if (curve.correlations.empty()) return DecayType::NO_SIGNAL;

        // Compute max absolute correlation
        float max_abs = 0.0f;
        for (float c : curve.correlations) {
            max_abs = std::max(max_abs, std::abs(c));
        }

        // No signal threshold
        if (max_abs < 0.05f) return DecayType::NO_SIGNAL;

        // Compute decay ratio: |corr at last horizon| / |corr at first horizon|
        float first_abs = std::abs(curve.correlations.front());
        float last_abs = std::abs(curve.correlations.back());

        if (first_abs < 1e-6f) {
            // First horizon is near zero — check if later is significant
            return (last_abs > 0.1f) ? DecayType::REGIME_INDICATOR : DecayType::NO_SIGNAL;
        }

        float decay_ratio = last_abs / first_abs;

        // Sharp decay: last is <30% of first → short-horizon signal
        if (decay_ratio < 0.3f) return DecayType::SHORT_HORIZON_SIGNAL;

        // Slow decay: last retains >30% of first → regime indicator
        return DecayType::REGIME_INDICATOR;
    }

    // Compute decay curves for a list of predictive features.
    std::vector<DecayCurve> compute_decay_for_predictive(
        const std::vector<BarFeatureRow>& rows,
        const std::vector<std::string>& predictive_features) const {

        std::vector<DecayCurve> curves;
        for (const auto& name : predictive_features) {
            curves.push_back(compute_decay(rows, name));
        }
        return curves;
    }

private:
    static float pearson_corr(const std::vector<float>& x, const std::vector<float>& y) {
        size_t n = std::min(x.size(), y.size());
        if (n < 3) return 0.0f;

        float sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(x[i]) || std::isnan(y[i])) continue;
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
        return std::max(-1.0f, std::min(1.0f, cov / denom));
    }
};
