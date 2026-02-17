#pragma once

// AnalysisResult — standard reporting struct for feature analysis results.
// Contains point estimate, CI, raw/corrected p-values, and flags.

struct AnalysisResult {
    float point_estimate = 0.0f;
    float ci_lower = 0.0f;
    float ci_upper = 0.0f;
    float raw_p_value = 1.0f;
    float corrected_p_value = 1.0f;
    bool survives_correction = false;
    bool is_suggestive = false;
    int sample_count = 0;
};
