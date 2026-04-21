use serde::{Deserialize, Serialize};

use crate::config::DecoderConfig;

/// Maps output neuron firing rate to a correction factor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decoder {
    pub scale: f64,
    pub epsilon: f64,
    pub scale_lr: f64,
}

impl Decoder {
    pub fn new(config: &DecoderConfig) -> Self {
        Self {
            scale: config.initial_scale,
            epsilon: config.epsilon,
            scale_lr: config.scale_lr,
        }
    }

    pub fn decode(&self, firing_rate: f64) -> f64 {
        let factor = (self.scale * (firing_rate - 0.5)).exp();
        factor.max(self.epsilon)
    }

    pub fn learn(&mut self, ratio: f64, firing_rate: f64) {
        let error = ratio - 1.0;
        let grad = error * (firing_rate - 0.5);
        self.scale += self.scale_lr * grad;
        self.scale = self.scale.max(0.1);
    }

    pub fn reset(&mut self, config: &DecoderConfig) {
        self.scale = config.initial_scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_decoder() -> Decoder {
        Decoder::new(&DecoderConfig::default())
    }

    #[test]
    fn test_midpoint_gives_unity() {
        let d = default_decoder();
        let factor = d.decode(0.5);
        assert!((factor - 1.0).abs() < 1e-10, "f=0.5 should give 1.0, got {factor}");
    }

    #[test]
    fn test_high_rate_increases_factor() {
        let d = default_decoder();
        let factor = d.decode(0.8);
        assert!(factor > 1.0, "f>0.5 should give factor>1.0, got {factor}");
    }

    #[test]
    fn test_low_rate_decreases_factor() {
        let d = default_decoder();
        let factor = d.decode(0.2);
        assert!(factor < 1.0, "f<0.5 should give factor<1.0, got {factor}");
    }

    #[test]
    fn test_symmetry() {
        let d = default_decoder();
        let high = d.decode(0.7);
        let low = d.decode(0.3);
        assert!((high * low - 1.0).abs() < 1e-10, "should be symmetric: {high} * {low}");
    }

    #[test]
    fn test_epsilon_bound() {
        let d = default_decoder();
        let factor = d.decode(0.0);
        assert!(factor >= d.epsilon, "factor should be >= epsilon");
    }

    #[test]
    fn test_unbounded_above() {
        let d = default_decoder();
        let factor = d.decode(1.0);
        assert!(factor > 2.0, "f=1.0 should give large factor, got {factor}");
    }

    #[test]
    fn test_learn_adjusts_scale() {
        let mut d = default_decoder();
        let scale_before = d.scale;
        d.learn(2.0, 0.6);
        assert!((d.scale - scale_before).abs() > 1e-10, "scale should change");
    }

    #[test]
    fn test_learn_no_change_at_unity_ratio() {
        let mut d = default_decoder();
        let scale_before = d.scale;
        d.learn(1.0, 0.6);
        assert!((d.scale - scale_before).abs() < 1e-10, "scale should not change at ratio=1");
    }

    #[test]
    fn test_reset() {
        let cfg = DecoderConfig::default();
        let mut d = Decoder::new(&cfg);
        d.scale = 999.0;
        d.reset(&cfg);
        assert!((d.scale - cfg.initial_scale).abs() < 1e-10);
    }
}
