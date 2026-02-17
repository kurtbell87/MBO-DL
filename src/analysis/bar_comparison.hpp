#pragma once

#include "analysis/multiple_comparison.hpp"
#include "analysis/statistical_tests.hpp"
#include "features/bar_features.hpp"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// BarTypeData — input data for one bar type configuration
// ---------------------------------------------------------------------------
struct BarTypeData {
    std::string name;
    std::vector<float> returns;
    std::vector<int> daily_bar_counts;
    float excess_mi_sum = 0.0f;
};

// ---------------------------------------------------------------------------
// BarComparisonResult — analysis output for one bar type
// ---------------------------------------------------------------------------
struct BarComparisonResult {
    std::string name;

    // Jarque-Bera
    float jarque_bera_stat = 0.0f;
    float jarque_bera_p = 1.0f;
    float jarque_bera_corrected_p = 1.0f;
    bool jarque_bera_survives = false;

    // ARCH LM
    float arch_lm_stat = 0.0f;
    float arch_lm_p = 1.0f;
    float arch_lm_corrected_p = 1.0f;
    bool arch_lm_survives = false;

    // ACF at lags 1, 5, 10
    std::vector<float> acf_values;

    // Ljung-Box at lags 1, 5, 10
    std::vector<TestResult> ljung_box_results;

    // AR R²
    float ar_r2 = 0.0f;

    // Aggregate excess MI
    float excess_mi_sum = 0.0f;

    // Bar count CV
    float bar_count_cv = 0.0f;

    // Warmup exclusion flag
    bool warmup_excluded = true;
};

// ---------------------------------------------------------------------------
// BarComparisonAnalyzer — computes all bar-type comparison metrics
// ---------------------------------------------------------------------------
class BarComparisonAnalyzer {
public:
    std::vector<BarComparisonResult> compare_bar_types(
        const std::vector<BarTypeData>& bar_types) {

        std::vector<BarComparisonResult> results;
        results.reserve(bar_types.size());

        // Compute raw metrics per bar type
        for (const auto& bt : bar_types) {
            BarComparisonResult r;
            r.name = bt.name;

            // Jarque-Bera
            auto jb = jarque_bera_test(bt.returns);
            r.jarque_bera_stat = jb.statistic;
            r.jarque_bera_p = jb.p_value;

            // ARCH LM
            auto arch = arch_lm_test(bt.returns);
            r.arch_lm_stat = arch.statistic;
            r.arch_lm_p = arch.p_value;

            // ACF of |return| at lags 1, 5, 10
            std::vector<float> abs_returns(bt.returns.size());
            for (size_t i = 0; i < bt.returns.size(); ++i) {
                abs_returns[i] = std::abs(bt.returns[i]);
            }
            r.acf_values = compute_acf(abs_returns, {1, 5, 10});

            // Ljung-Box at lags 1, 5, 10
            r.ljung_box_results = ljung_box_test(bt.returns, {1, 5, 10});

            // AR R²
            r.ar_r2 = compute_ar_r2(bt.returns, 10);
            if (std::isnan(r.ar_r2)) r.ar_r2 = 0.0f;

            // Excess MI sum
            r.excess_mi_sum = bt.excess_mi_sum;

            // Bar count CV
            r.bar_count_cv = compute_bar_count_cv(bt.daily_bar_counts);

            // Warmup exclusion flag (always true — we only analyze non-warmup data)
            r.warmup_excluded = true;

            results.push_back(r);
        }

        // Apply Holm-Bonferroni within each metric family
        apply_corrections(results);

        return results;
    }

private:
    void apply_corrections(std::vector<BarComparisonResult>& results) {
        size_t m = results.size();
        if (m <= 1) {
            for (auto& r : results) {
                r.jarque_bera_corrected_p = r.jarque_bera_p;
                r.arch_lm_corrected_p = r.arch_lm_p;
                r.jarque_bera_survives = (r.jarque_bera_corrected_p < 0.05f);
                r.arch_lm_survives = (r.arch_lm_corrected_p < 0.05f);
            }
            return;
        }

        // Collect raw p-values per family
        std::vector<float> jb_pvals, arch_pvals;
        for (const auto& r : results) {
            jb_pvals.push_back(r.jarque_bera_p);
            arch_pvals.push_back(r.arch_lm_p);
        }

        auto jb_corrected = holm_bonferroni_correct(jb_pvals);
        auto arch_corrected = holm_bonferroni_correct(arch_pvals);

        for (size_t i = 0; i < m; ++i) {
            results[i].jarque_bera_corrected_p = jb_corrected[i];
            results[i].arch_lm_corrected_p = arch_corrected[i];
            results[i].jarque_bera_survives = (jb_corrected[i] < 0.05f);
            results[i].arch_lm_survives = (arch_corrected[i] < 0.05f);
        }
    }
};
