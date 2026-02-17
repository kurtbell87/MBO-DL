#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

// Holm-Bonferroni step-down correction for multiple comparisons.
// Returns corrected p-values in the SAME order as input.
inline std::vector<float> holm_bonferroni_correct(const std::vector<float>& raw_pvals) {
    size_t m = raw_pvals.size();
    if (m == 0) return {};
    if (m == 1) return raw_pvals;

    // Create index array sorted by ascending raw p-value
    std::vector<size_t> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return raw_pvals[a] < raw_pvals[b];
    });

    // Compute corrected p-values in sorted order
    std::vector<float> corrected(m);
    float running_max = 0.0f;
    for (size_t rank = 0; rank < m; ++rank) {
        size_t idx = order[rank];
        float adjusted = raw_pvals[idx] * static_cast<float>(m - rank);
        adjusted = std::min(adjusted, 1.0f);
        // Enforce monotonicity: corrected p-values must be non-decreasing in sorted order
        running_max = std::max(running_max, adjusted);
        corrected[idx] = running_max;
    }

    return corrected;
}
