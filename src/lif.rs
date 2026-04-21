use serde::{Deserialize, Serialize};

use crate::config;

const V_REST: f64 = 0.0;

/// Configurable neuron parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronConfig {
    pub v_threshold: f64,
    pub v_reset: f64,
    pub refractory_steps: u8,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            v_threshold: config::DEFAULT_V_THRESHOLD,
            v_reset: config::DEFAULT_V_RESET,
            refractory_steps: config::DEFAULT_REFRACTORY_STEPS,
        }
    }
}

/// Leaky Integrate-and-Fire neuron.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifNeuron {
    pub v: f64,
    pub tau: f64,
    pub refractory: u8,
    pub spiked: bool,
    pub spike_count: u32,
}

impl LifNeuron {
    pub fn new(tau: f64) -> Self {
        Self {
            v: V_REST,
            tau,
            refractory: 0,
            spiked: false,
            spike_count: 0,
        }
    }

    pub fn begin_tick(&mut self) {
        self.spike_count = 0;
    }

    pub fn step(&mut self, current: f64, config: &NeuronConfig) {
        self.spiked = false;

        if self.refractory > 0 {
            self.refractory -= 1;
            return;
        }

        let dv = (-(self.v - V_REST) / self.tau) + current;
        self.v += dv;

        if self.v >= config.v_threshold {
            self.spiked = true;
            self.spike_count += 1;
            self.v = config.v_reset;
            self.refractory = config.refractory_steps;
        }
    }

    pub fn firing_rate(&self, steps_per_tick: u32) -> f64 {
        self.spike_count as f64 / steps_per_tick as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_neuron_at_rest() {
        let n = LifNeuron::new(10.0);
        assert_eq!(n.v, 0.0);
        assert_eq!(n.tau, 10.0);
        assert_eq!(n.refractory, 0);
        assert!(!n.spiked);
        assert_eq!(n.spike_count, 0);
    }

    #[test]
    fn test_fires_above_threshold() {
        let config = NeuronConfig::default();
        let mut n = LifNeuron::new(10.0);
        for _ in 0..20 {
            n.step(2.0, &config);
        }
        assert!(n.spike_count > 0);
    }

    #[test]
    fn test_no_fire_below_threshold() {
        let config = NeuronConfig::default();
        let mut n = LifNeuron::new(10.0);
        for _ in 0..20 {
            n.step(0.01, &config);
        }
        assert_eq!(n.spike_count, 0);
    }

    #[test]
    fn test_refractory_period() {
        let config = NeuronConfig { refractory_steps: 3, ..Default::default() };
        let mut n = LifNeuron::new(10.0);
        for _ in 0..50 {
            n.step(5.0, &config);
        }
        assert!(n.spike_count <= 13);
        assert!(n.spike_count > 0);
    }

    #[test]
    fn test_begin_tick_resets_count() {
        let config = NeuronConfig::default();
        let mut n = LifNeuron::new(10.0);
        for _ in 0..20 {
            n.step(5.0, &config);
        }
        assert!(n.spike_count > 0);
        n.begin_tick();
        assert_eq!(n.spike_count, 0);
    }

    #[test]
    fn test_firing_rate() {
        let mut n = LifNeuron::new(10.0);
        n.spike_count = 5;
        assert!((n.firing_rate(20) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_tau_decay() {
        let config = NeuronConfig::default();
        let mut slow = LifNeuron::new(100.0);
        let mut fast = LifNeuron::new(3.0);
        let current = 0.5;
        for _ in 0..20 {
            slow.step(current, &config);
            fast.step(current, &config);
        }
        assert!(slow.spike_count >= fast.spike_count);
    }
}
