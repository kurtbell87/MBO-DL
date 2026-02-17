#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// TestResult — generic {statistic, p_value} pair
// ---------------------------------------------------------------------------
struct TestResult {
    float statistic = 0.0f;
    float p_value = 1.0f;
};

// ---------------------------------------------------------------------------
// Chi-squared CDF approximation (for 2 degrees of freedom: JB, and general)
// Uses the regularized lower incomplete gamma function approximation.
// For integer/small df, chi2 CDF = 1 - exp(-x/2) * sum of terms.
// ---------------------------------------------------------------------------
namespace detail {

// Approximate chi-squared survival function (1 - CDF) for df degrees of freedom.
// Uses a series expansion of the regularized incomplete gamma function.
inline float chi2_sf(float x, int df) {
    if (x <= 0.0f) return 1.0f;
    // Use the series expansion: P(a, x) = e^(-x) * x^a * sum(x^k / Gamma(a+k+1))
    // Q(a, x) = 1 - P(a, x)
    float a = static_cast<float>(df) / 2.0f;
    float half_x = x / 2.0f;

    // For small df, use direct computation
    if (df == 2) {
        return std::exp(-half_x);
    }

    // General case: use the upper incomplete gamma approximation
    // via continued fraction or series
    // Simple series: Q(a, x) ≈ e^(-half_x) * half_x^a / Gamma(a) * sum...
    // Use a practical iterative approach
    float sum = 1.0f;
    float term = 1.0f;
    for (int k = 1; k < 200; ++k) {
        term *= half_x / (a + static_cast<float>(k));
        sum += term;
        if (term < 1e-8f) break;
    }

    // P(a, half_x) = e^(-half_x) * half_x^a * sum / Gamma(a+1)
    // We need log to avoid overflow
    float log_p = -half_x + a * std::log(half_x) - std::lgamma(a + 1.0f) + std::log(sum);
    float p = std::exp(log_p);
    p = std::min(1.0f, std::max(0.0f, p));
    return 1.0f - p;
}

// Normal CDF approximation (Abramowitz & Stegun)
inline float normal_cdf(float x) {
    // Using the error function
    return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

// Normal quantile (inverse CDF) — rational approximation
inline float normal_quantile(float p) {
    if (p <= 0.0f) return -1e10f;
    if (p >= 1.0f) return 1e10f;
    if (std::abs(p - 0.5f) < 1e-10f) return 0.0f;

    // Rational approximation (Beasley-Springer-Moro)
    float t;
    if (p < 0.5f) {
        t = std::sqrt(-2.0f * std::log(p));
    } else {
        t = std::sqrt(-2.0f * std::log(1.0f - p));
    }

    // Coefficients for rational approximation
    float c0 = 2.515517f, c1 = 0.802853f, c2 = 0.010328f;
    float d1 = 1.432788f, d2 = 0.189269f, d3 = 0.001308f;

    float result = t - (c0 + c1 * t + c2 * t * t) /
                       (1.0f + d1 * t + d2 * t * t + d3 * t * t * t);
    if (p < 0.5f) result = -result;
    return result;
}

// t-distribution CDF approximation for large df using normal approximation
inline float t_cdf(float t, int df) {
    if (df > 30) {
        return normal_cdf(t);
    }
    // For smaller df, use the incomplete beta function approximation
    float x = static_cast<float>(df) / (static_cast<float>(df) + t * t);
    // Two-tailed: P(T > |t|) using regularized incomplete beta
    // Simple approximation for moderate df
    float a = static_cast<float>(df) / 2.0f;
    float b = 0.5f;

    // Use the normal approximation with correction
    float g = std::sqrt(2.0f / (9.0f * a)) * (std::pow(x, 1.0f/3.0f) -
              (1.0f - 2.0f / (9.0f * a)));
    return normal_cdf(-std::abs(t)) * 2.0f; // crude two-tailed
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Jarque-Bera normality test
// JB = n/6 * (S² + K²/4) where S = skewness, K = excess kurtosis
// Under H₀ (normality), JB ~ χ²(2)
// ---------------------------------------------------------------------------
inline TestResult jarque_bera_test(const std::vector<float>& data) {
    // Filter NaN
    std::vector<float> clean;
    clean.reserve(data.size());
    for (float v : data) {
        if (!std::isnan(v)) clean.push_back(v);
    }

    size_t n = clean.size();
    if (n < 3) {
        return {0.0f, 1.0f};
    }

    // Compute mean
    float sum = 0.0f;
    for (float v : clean) sum += v;
    float mean = sum / static_cast<float>(n);

    // Compute central moments
    float m2 = 0.0f, m3 = 0.0f, m4 = 0.0f;
    for (float v : clean) {
        float d = v - mean;
        float d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    float nf = static_cast<float>(n);
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;

    if (m2 < 1e-12f) {
        // Zero variance → undefined skewness/kurtosis, return 0
        return {0.0f, 1.0f};
    }

    float skewness = m3 / std::pow(m2, 1.5f);
    float kurtosis = m4 / (m2 * m2) - 3.0f;  // excess kurtosis

    float jb = nf / 6.0f * (skewness * skewness + kurtosis * kurtosis / 4.0f);
    float p = detail::chi2_sf(jb, 2);

    return {jb, p};
}

// ---------------------------------------------------------------------------
// ARCH LM test for heteroskedasticity
// Regress squared residuals on lagged squared residuals.
// Test statistic: n*R² ~ χ²(p) where p = number of lags.
// ---------------------------------------------------------------------------
inline TestResult arch_lm_test(const std::vector<float>& returns, int lags = 1) {
    size_t n = returns.size();
    if (n < static_cast<size_t>(lags + 2)) {
        return {0.0f, 1.0f};
    }

    // Use double precision throughout to handle extreme values
    double d_mean = 0.0;
    for (float r : returns) d_mean += static_cast<double>(r);
    d_mean /= static_cast<double>(n);

    // Squared residuals in double precision
    std::vector<double> sq(n);
    for (size_t i = 0; i < n; ++i) {
        double e = static_cast<double>(returns[i]) - d_mean;
        sq[i] = e * e;
    }

    // Regress sq[t] on sq[t-1] using OLS in double precision
    size_t T = n - static_cast<size_t>(lags);
    double y_sum = 0, x_sum = 0, xy_sum = 0, xx_sum = 0, yy_sum = 0;
    for (size_t t = static_cast<size_t>(lags); t < n; ++t) {
        double y = sq[t];
        double x = sq[t - 1];
        y_sum += y;
        x_sum += x;
        xy_sum += x * y;
        xx_sum += x * x;
        yy_sum += y * y;
    }
    double Tf = static_cast<double>(T);
    double denom = Tf * xx_sum - x_sum * x_sum;
    if (std::abs(denom) < 1e-30) {
        return {0.0f, 1.0f};
    }

    // R² = (correlation)²
    double cov_xy = Tf * xy_sum - x_sum * y_sum;
    double var_y = Tf * yy_sum - y_sum * y_sum;
    double r2 = (var_y > 1e-30) ? (cov_xy * cov_xy) / (denom * var_y) : 0.0;
    r2 = std::max(0.0, std::min(1.0, r2));

    float stat = static_cast<float>(Tf * r2);
    float p = detail::chi2_sf(stat, lags);

    return {stat, p};
}

// ---------------------------------------------------------------------------
// ACF — autocorrelation function at specified lags
// ---------------------------------------------------------------------------
inline std::vector<float> compute_acf(const std::vector<float>& data,
                                       const std::vector<int>& lags) {
    size_t n = data.size();
    std::vector<float> result;
    result.reserve(lags.size());

    if (n < 2) {
        for (size_t i = 0; i < lags.size(); ++i) result.push_back(0.0f);
        return result;
    }

    // Compute mean
    float sum = 0.0f;
    for (float v : data) sum += v;
    float mean = sum / static_cast<float>(n);

    // Compute variance (denominator: gamma_0)
    float gamma0 = 0.0f;
    for (float v : data) {
        float d = v - mean;
        gamma0 += d * d;
    }
    if (gamma0 < 1e-12f) {
        for (size_t i = 0; i < lags.size(); ++i) result.push_back(0.0f);
        return result;
    }

    for (int lag : lags) {
        if (lag == 0) {
            result.push_back(1.0f);
            continue;
        }
        if (static_cast<size_t>(lag) >= n) {
            result.push_back(0.0f);
            continue;
        }
        float gamma_k = 0.0f;
        for (size_t i = static_cast<size_t>(lag); i < n; ++i) {
            gamma_k += (data[i] - mean) * (data[i - lag] - mean);
        }
        result.push_back(gamma_k / gamma0);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Ljung-Box test at specified lags
// Q = n(n+2) * sum_{k=1}^{h} (r_k² / (n-k))
// Under H₀: Q ~ χ²(h)
// ---------------------------------------------------------------------------
inline std::vector<TestResult> ljung_box_test(const std::vector<float>& data,
                                               const std::vector<int>& lags) {
    size_t n = data.size();
    std::vector<TestResult> results;

    if (n < 3) {
        for (size_t i = 0; i < lags.size(); ++i) results.push_back({0.0f, 1.0f});
        return results;
    }

    // Compute mean
    float sum = 0.0f;
    for (float v : data) sum += v;
    float mean = sum / static_cast<float>(n);

    // Compute gamma_0
    float gamma0 = 0.0f;
    for (float v : data) {
        float d = v - mean;
        gamma0 += d * d;
    }

    if (gamma0 < 1e-12f) {
        for (size_t i = 0; i < lags.size(); ++i) results.push_back({0.0f, 1.0f});
        return results;
    }

    float nf = static_cast<float>(n);

    for (int max_lag : lags) {
        float Q = 0.0f;
        for (int k = 1; k <= max_lag && static_cast<size_t>(k) < n; ++k) {
            float gamma_k = 0.0f;
            for (size_t i = static_cast<size_t>(k); i < n; ++i) {
                gamma_k += (data[i] - mean) * (data[i - k] - mean);
            }
            float rho_k = gamma_k / gamma0;
            Q += (rho_k * rho_k) / (nf - static_cast<float>(k));
        }
        Q *= nf * (nf + 2.0f);
        float p = detail::chi2_sf(Q, max_lag);
        results.push_back({Q, p});
    }

    return results;
}

// ---------------------------------------------------------------------------
// AR R² — fit AR(p) model via OLS and return R²
// ---------------------------------------------------------------------------
inline float compute_ar_r2(const std::vector<float>& data, int p) {
    int n = static_cast<int>(data.size());
    if (n <= p + 1) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Build design matrix and target vector
    int T = n - p;
    // Use simple OLS: y = X * beta
    // y[t] = data[p + t], X[t] = [data[p+t-1], data[p+t-2], ..., data[t]]

    // Compute y mean
    float y_mean = 0.0f;
    for (int t = 0; t < T; ++t) {
        y_mean += data[p + t];
    }
    y_mean /= static_cast<float>(T);

    // Total sum of squares
    float ss_tot = 0.0f;
    for (int t = 0; t < T; ++t) {
        float d = data[p + t] - y_mean;
        ss_tot += d * d;
    }
    if (ss_tot < 1e-12f) return 0.0f;

    // For AR models, use a simple iterative least squares approach
    // For AR(p) with p up to ~10, use the Yule-Walker equations (simpler)
    // or direct OLS via normal equations: X'X * beta = X'y

    // Build X'X (p x p) and X'y (p x 1)
    std::vector<float> XtX(p * p, 0.0f);
    std::vector<float> Xty(p, 0.0f);

    for (int t = 0; t < T; ++t) {
        float y = data[p + t];
        for (int i = 0; i < p; ++i) {
            float xi = data[p + t - 1 - i];
            Xty[i] += xi * y;
            for (int j = 0; j < p; ++j) {
                float xj = data[p + t - 1 - j];
                XtX[i * p + j] += xi * xj;
            }
        }
    }

    // Solve via Cholesky or Gaussian elimination
    // Simple Gaussian elimination with partial pivoting
    std::vector<float> A(p * (p + 1));
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            A[i * (p + 1) + j] = XtX[i * p + j];
        }
        A[i * (p + 1) + p] = Xty[i];
    }

    // Forward elimination
    for (int col = 0; col < p; ++col) {
        // Find pivot
        int max_row = col;
        float max_val = std::abs(A[col * (p + 1) + col]);
        for (int row = col + 1; row < p; ++row) {
            float val = std::abs(A[row * (p + 1) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }
        if (max_val < 1e-12f) return 0.0f;  // Singular

        // Swap rows
        if (max_row != col) {
            for (int j = 0; j <= p; ++j) {
                std::swap(A[col * (p + 1) + j], A[max_row * (p + 1) + j]);
            }
        }

        // Eliminate
        for (int row = col + 1; row < p; ++row) {
            float factor = A[row * (p + 1) + col] / A[col * (p + 1) + col];
            for (int j = col; j <= p; ++j) {
                A[row * (p + 1) + j] -= factor * A[col * (p + 1) + j];
            }
        }
    }

    // Back substitution
    std::vector<float> beta(p, 0.0f);
    for (int i = p - 1; i >= 0; --i) {
        float sum = A[i * (p + 1) + p];
        for (int j = i + 1; j < p; ++j) {
            sum -= A[i * (p + 1) + j] * beta[j];
        }
        beta[i] = sum / A[i * (p + 1) + i];
    }

    // Compute predicted values and residual sum of squares
    float ss_res = 0.0f;
    for (int t = 0; t < T; ++t) {
        float y_pred = 0.0f;
        for (int i = 0; i < p; ++i) {
            y_pred += beta[i] * data[p + t - 1 - i];
        }
        float resid = data[p + t] - y_pred;
        ss_res += resid * resid;
    }

    float r2 = 1.0f - ss_res / ss_tot;
    return std::max(0.0f, std::min(1.0f, r2));
}

// ---------------------------------------------------------------------------
// CV of daily bar counts (coefficient of variation = std / mean)
// ---------------------------------------------------------------------------
inline float compute_bar_count_cv(const std::vector<int>& daily_counts) {
    if (daily_counts.empty()) return 0.0f;
    if (daily_counts.size() == 1) return 0.0f;

    float sum = 0.0f;
    for (int c : daily_counts) sum += static_cast<float>(c);
    float mean = sum / static_cast<float>(daily_counts.size());
    if (mean < 1e-12f) return 0.0f;

    float var = 0.0f;
    for (int c : daily_counts) {
        float d = static_cast<float>(c) - mean;
        var += d * d;
    }
    var /= static_cast<float>(daily_counts.size());

    return std::sqrt(var) / mean;
}

// ---------------------------------------------------------------------------
// Aggregate MI — sum of all excess MI values (including negative)
// ---------------------------------------------------------------------------
inline float compute_aggregate_mi(const std::vector<float>& excess_mis) {
    float total = 0.0f;
    for (float v : excess_mis) total += v;
    return total;
}
