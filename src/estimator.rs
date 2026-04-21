use std::time::Duration;

use crate::config;

/// Pluggable base ETA estimator.
///
/// Implement this trait to provide a custom baseline for the RSNN correction factor.
/// The default implementation is [`EmaEstimator`], which uses an exponential moving
/// average of per-step duration.
pub trait BaseEstimator: Send + Sync {
    /// Feed a new progress observation.
    fn update(&mut self, position: u64, length: u64, elapsed: Duration);
    /// Return the estimated remaining time, or `None` if not ready.
    fn estimate(&self) -> Option<Duration>;
    /// Whether the estimator has seen enough data to produce useful estimates.
    fn is_warm(&self) -> bool;
    /// Reset all internal state.
    fn reset(&mut self);
    /// Current estimated processing rate (steps per second).
    fn steps_per_sec(&self) -> f64;
    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn BaseEstimator>;
}

/// EMA-based estimator tracking exponentially weighted step duration.
///
/// Computes `ETA = remaining_steps * ema_step_duration` where the step duration
/// is smoothed with an exponential moving average (configurable alpha).
/// Returns `None` until `warmup_ticks` observations have been collected.
pub struct EmaEstimator {
    alpha: f64,
    warmup_ticks: u64,
    tick_count: u64,
    ema_step_duration: f64,
    last_position: u64,
    last_elapsed: f64,
    current_length: u64,
    current_position: u64,
}

impl EmaEstimator {
    pub fn new(alpha: f64, warmup_ticks: u64) -> Self {
        Self {
            alpha,
            warmup_ticks,
            tick_count: 0,
            ema_step_duration: 0.0,
            last_position: 0,
            last_elapsed: 0.0,
            current_length: 0,
            current_position: 0,
        }
    }
}

impl Default for EmaEstimator {
    fn default() -> Self {
        Self::new(config::DEFAULT_EMA_ALPHA, config::DEFAULT_EMA_WARMUP)
    }
}

impl Clone for EmaEstimator {
    fn clone(&self) -> Self {
        Self {
            alpha: self.alpha,
            warmup_ticks: self.warmup_ticks,
            tick_count: self.tick_count,
            ema_step_duration: self.ema_step_duration,
            last_position: self.last_position,
            last_elapsed: self.last_elapsed,
            current_length: self.current_length,
            current_position: self.current_position,
        }
    }
}

impl BaseEstimator for EmaEstimator {
    fn update(&mut self, position: u64, length: u64, elapsed: Duration) {
        let elapsed_secs = elapsed.as_secs_f64();
        self.current_length = length;
        self.current_position = position;

        if self.tick_count > 0 {
            let dt = elapsed_secs - self.last_elapsed;
            let dp = position.saturating_sub(self.last_position);
            if dp > 0 && dt > 0.0 {
                let step_dur = dt / dp as f64;
                if self.ema_step_duration == 0.0 {
                    self.ema_step_duration = step_dur;
                } else {
                    self.ema_step_duration = self.alpha * step_dur
                        + (1.0 - self.alpha) * self.ema_step_duration;
                }
            }
        }

        self.last_position = position;
        self.last_elapsed = elapsed_secs;
        self.tick_count += 1;
    }

    fn estimate(&self) -> Option<Duration> {
        if !self.is_warm() || self.current_length == 0 {
            return None;
        }
        let remaining = self.current_length.saturating_sub(self.current_position);
        let eta_secs = remaining as f64 * self.ema_step_duration;
        Some(Duration::from_secs_f64(eta_secs.max(0.0)))
    }

    fn is_warm(&self) -> bool {
        self.tick_count > self.warmup_ticks
    }

    fn reset(&mut self) {
        self.tick_count = 0;
        self.ema_step_duration = 0.0;
        self.last_position = 0;
        self.last_elapsed = 0.0;
        self.current_length = 0;
        self.current_position = 0;
    }

    fn steps_per_sec(&self) -> f64 {
        if self.ema_step_duration > 0.0 {
            1.0 / self.ema_step_duration
        } else {
            0.0
        }
    }

    fn clone_box(&self) -> Box<dyn BaseEstimator> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_warm_initially() {
        let est = EmaEstimator::default();
        assert!(!est.is_warm());
        assert!(est.estimate().is_none());
    }

    #[test]
    fn test_warms_up_after_ticks() {
        let mut est = EmaEstimator::new(0.05, 3);
        for i in 1..=4 {
            est.update(i * 10, 100, Duration::from_secs(i));
        }
        assert!(est.is_warm());
        assert!(est.estimate().is_some());
    }

    #[test]
    fn test_estimate_constant_rate() {
        let mut est = EmaEstimator::new(0.5, 2);
        for i in 1..=20 {
            est.update(i * 10, 1000, Duration::from_millis(i * 100));
        }
        let eta = est.estimate().unwrap();
        let eta_secs = eta.as_secs_f64();
        assert!(eta_secs > 5.0 && eta_secs < 12.0,
            "expected ETA near 8s, got {eta_secs}s");
    }

    #[test]
    fn test_estimate_zero_length() {
        let mut est = EmaEstimator::new(0.5, 1);
        est.update(10, 0, Duration::from_secs(1));
        est.update(20, 0, Duration::from_secs(2));
        assert!(est.estimate().is_none());
    }

    #[test]
    fn test_estimate_completed() {
        let mut est = EmaEstimator::new(0.5, 1);
        est.update(50, 100, Duration::from_secs(1));
        est.update(100, 100, Duration::from_secs(2));
        let eta = est.estimate().unwrap();
        assert!(eta.as_secs_f64() < 0.01, "should be near 0 when complete");
    }

    #[test]
    fn test_steps_per_sec() {
        let mut est = EmaEstimator::new(0.5, 1);
        est.update(0, 100, Duration::from_secs(0));
        est.update(100, 100, Duration::from_secs(1));
        let sps = est.steps_per_sec();
        assert!(sps > 50.0 && sps < 200.0, "expected ~100 sps, got {sps}");
    }

    #[test]
    fn test_reset() {
        let mut est = EmaEstimator::new(0.5, 2);
        est.update(50, 100, Duration::from_secs(1));
        est.update(60, 100, Duration::from_secs(2));
        est.reset();
        assert!(!est.is_warm());
        assert!(est.estimate().is_none());
    }

    #[test]
    fn test_clone_box() {
        let est = EmaEstimator::default();
        let cloned = est.clone_box();
        assert!(!cloned.is_warm());
    }
}
