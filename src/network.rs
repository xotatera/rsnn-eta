use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::config::NetworkConfig;
use crate::lif::LifNeuron;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Synapse {
    pub target: usize,
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnnNetwork {
    pub neurons: Vec<LifNeuron>,
    pub input_synapses: Vec<Vec<(usize, f64)>>,
    pub recurrent_synapses: Vec<Vec<Synapse>>,
    pub is_excitatory: Vec<bool>,
    pub input_dim: usize,
    pub num_neurons: usize,
}

impl SnnNetwork {
    pub fn new(input_dim: usize, config: &NetworkConfig, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let n = config.num_neurons;

        let neurons: Vec<LifNeuron> = (0..n)
            .map(|_| {
                let log_tau = rng.gen_range(config.tau_min.ln()..config.tau_max.ln());
                LifNeuron::new(log_tau.exp())
            })
            .collect();

        let n_exc = (n as f64 * config.excitatory_frac) as usize;
        let is_excitatory: Vec<bool> = (0..n).map(|i| i < n_exc).collect();

        let input_synapses = Self::init_input_synapses(
            input_dim, n, config.input_sparsity, config.init_input_scale, &mut rng,
        );
        let recurrent_synapses = Self::init_recurrent_synapses(
            n, &is_excitatory, config.recurrent_sparsity, config.init_recurrent_scale, &mut rng,
        );

        Self { neurons, input_synapses, recurrent_synapses, is_excitatory, input_dim, num_neurons: n }
    }

    fn init_input_synapses(
        input_dim: usize, num_neurons: usize, sparsity: f64, scale: f64, rng: &mut SmallRng,
    ) -> Vec<Vec<(usize, f64)>> {
        (0..num_neurons)
            .map(|_| {
                let mut syns = Vec::new();
                for feat in 0..input_dim {
                    if rng.gen::<f64>() < sparsity {
                        syns.push((feat, rng.gen_range(-scale..scale)));
                    }
                }
                syns
            })
            .collect()
    }

    fn init_recurrent_synapses(
        num_neurons: usize, is_excitatory: &[bool], sparsity: f64, scale: f64, rng: &mut SmallRng,
    ) -> Vec<Vec<Synapse>> {
        (0..num_neurons)
            .map(|pre| {
                let mut syns = Vec::new();
                for post in 0..num_neurons {
                    if post != pre && rng.gen::<f64>() < sparsity {
                        let w = rng.gen_range(0.0..scale);
                        let weight = if is_excitatory[pre] { w } else { -w };
                        syns.push(Synapse { target: post, weight });
                    }
                }
                syns
            })
            .collect()
    }

    pub fn output_neuron(&self) -> usize {
        self.num_neurons - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> NetworkConfig {
        NetworkConfig::default()
    }

    #[test]
    fn test_network_dimensions() {
        let cfg = default_config();
        let net = SnnNetwork::new(4, &cfg, 42);
        assert_eq!(net.neurons.len(), cfg.num_neurons);
        assert_eq!(net.input_synapses.len(), cfg.num_neurons);
        assert_eq!(net.recurrent_synapses.len(), cfg.num_neurons);
        assert_eq!(net.is_excitatory.len(), cfg.num_neurons);
        assert_eq!(net.input_dim, 4);
    }

    #[test]
    fn test_excitatory_inhibitory_ratio() {
        let cfg = default_config();
        let net = SnnNetwork::new(4, &cfg, 42);
        let n_exc = net.is_excitatory.iter().filter(|&&e| e).count();
        let expected = (cfg.num_neurons as f64 * cfg.excitatory_frac) as usize;
        assert_eq!(n_exc, expected);
    }

    #[test]
    fn test_no_self_connections() {
        let cfg = default_config();
        let net = SnnNetwork::new(4, &cfg, 42);
        for (pre, syns) in net.recurrent_synapses.iter().enumerate() {
            for syn in syns {
                assert_ne!(syn.target, pre, "self-connection found at neuron {pre}");
            }
        }
    }

    #[test]
    fn test_weight_sign_constraints() {
        let cfg = default_config();
        let net = SnnNetwork::new(4, &cfg, 42);
        for (pre, syns) in net.recurrent_synapses.iter().enumerate() {
            for syn in syns {
                if net.is_excitatory[pre] {
                    assert!(syn.weight >= 0.0, "excitatory neuron {pre} has negative weight");
                } else {
                    assert!(syn.weight <= 0.0, "inhibitory neuron {pre} has positive weight");
                }
            }
        }
    }

    #[test]
    fn test_input_sparsity_approximate() {
        let cfg = default_config();
        let net = SnnNetwork::new(4, &cfg, 42);
        let total_possible = cfg.num_neurons * 4;
        let actual: usize = net.input_synapses.iter().map(|s| s.len()).sum();
        let ratio = actual as f64 / total_possible as f64;
        assert!((ratio - cfg.input_sparsity).abs() < 0.15,
            "input sparsity {ratio} far from target {}", cfg.input_sparsity);
    }

    #[test]
    fn test_output_neuron_index() {
        let cfg = default_config();
        let net = SnnNetwork::new(4, &cfg, 42);
        assert_eq!(net.output_neuron(), cfg.num_neurons - 1);
    }
}
