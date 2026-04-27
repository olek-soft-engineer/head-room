//! Numeric statistics helpers — port of Python's `statistics` module
//! semantics used by `SmartAnalyzer`.
//!
//! Python's `statistics` module uses **sample** variance/stdev (n-1
//! denominator), not population (n denominator). Mismatching the
//! denominator silently shifts every variance-based decision (change
//! points, anomaly thresholds, crushability cases). These helpers
//! mirror Python's defaults.

/// Arithmetic mean. Returns `None` on empty input — Python's
/// `statistics.mean([])` raises `StatisticsError`; we model that as
/// "no value to return", and callers must handle it.
///
/// Also returns `None` if the result is non-finite (Inf/NaN). Python's
/// numeric-stats path in `_analyze_field` is wrapped in
/// `try/except (OverflowError, ValueError)` and resets all stats to
/// `None` on failure; mirroring that here keeps parity on extreme floats.
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let sum: f64 = values.iter().sum();
    let m = sum / values.len() as f64;
    if m.is_finite() {
        Some(m)
    } else {
        None
    }
}

/// Sample variance with `n-1` denominator (Python `statistics.variance`).
/// Requires at least 2 values; returns `None` for fewer (mirrors
/// Python which raises `StatisticsError` for n < 2). Also returns `None`
/// on non-finite results — see `mean` for rationale.
pub fn sample_variance(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let m = mean(values)?;
    let sum_sq_diff: f64 = values.iter().map(|v| (v - m).powi(2)).sum();
    let var = sum_sq_diff / (values.len() - 1) as f64;
    if var.is_finite() {
        Some(var)
    } else {
        None
    }
}

/// Sample standard deviation — sqrt of `sample_variance`. Same n>=2
/// requirement as the variance helper. `None` propagates from
/// `sample_variance`, including on non-finite inputs.
pub fn sample_stdev(values: &[f64]) -> Option<f64> {
    sample_variance(values).map(f64::sqrt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn mean_empty_is_none() {
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn mean_single() {
        assert!(approx_eq(mean(&[5.0]).unwrap(), 5.0));
    }

    #[test]
    fn mean_basic() {
        // Python: statistics.mean([1, 2, 3, 4, 5]) == 3.0
        assert!(approx_eq(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(), 3.0));
    }

    #[test]
    fn sample_variance_too_few_values_is_none() {
        // Python: statistics.variance([5]) raises; we return None.
        assert_eq!(sample_variance(&[]), None);
        assert_eq!(sample_variance(&[5.0]), None);
    }

    #[test]
    fn sample_variance_uses_n_minus_1_denominator() {
        // Python: statistics.variance([1, 2, 3, 4, 5]) == 2.5
        // (Population variance with n in denominator would give 2.0.)
        let v = sample_variance(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(approx_eq(v, 2.5), "got {v}, expected 2.5");
    }

    #[test]
    fn sample_stdev_basic() {
        // Python: statistics.stdev([1, 2, 3, 4, 5]) == sqrt(2.5)
        let s = sample_stdev(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(approx_eq(s, 2.5_f64.sqrt()), "got {s}");
    }

    #[test]
    fn sample_variance_constant_values_is_zero() {
        // All-identical values: variance = 0.
        let v = sample_variance(&[7.0, 7.0, 7.0]).unwrap();
        assert!(approx_eq(v, 0.0));
    }

    #[test]
    fn mean_non_finite_overflow_returns_none() {
        // Python parity: extreme inputs that overflow during sum cause
        // `statistics.mean` to raise OverflowError, which `_analyze_field`
        // treats as "no stats". We mirror by returning None.
        let huge = f64::MAX / 2.0;
        let nums = vec![huge, huge, huge, huge];
        // Sum overflows to +Inf, mean = Inf — non-finite, must be None.
        assert_eq!(mean(&nums), None);
    }

    #[test]
    fn sample_variance_non_finite_returns_none() {
        // Squared-diff sum overflows. Mirrors Python's OverflowError path.
        let huge = 1e200;
        let v = sample_variance(&[huge, -huge]);
        // (1e200 - 0)² + (-1e200 - 0)² overflows → variance is Inf → None.
        assert_eq!(v, None);
    }

    #[test]
    fn sample_stdev_non_finite_returns_none() {
        let huge = 1e200;
        assert_eq!(sample_stdev(&[huge, -huge]), None);
    }
}
