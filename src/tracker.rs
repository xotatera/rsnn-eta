use std::time::{Duration, Instant};

use crate::config::{self, DecoderConfig, NetworkConfig, StdpConfig};
use crate::decoder::Decoder;
use crate::encoder::Encoder;
use crate::estimator::BaseEstimator;
use crate::network::SnnNetwork;
use crate::stdp::StdpState;

/// Core ETA engine: orchestrates encoder → RSNN → decoder → base estimator → correction.
pub struct RsnnEtaCore {
    pub network: SnnNetwork,
    pub stdp: StdpState,
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub base_estimator: Box<dyn BaseEstimator>,
    pub net_config: NetworkConfig,

    pub(crate) tick_count: u64,
    burn_in_ticks: u64,
    last_predicted_step_dt: Option<f64>,
    last_tick_time: Option<Instant>,

    ratio_mean: f64,
    ratio_var: f64,
    confidence_alpha: f64,
    confidence: f64,

    pub last_eta: Option<Duration>,
}

impl RsnnEtaCore {
    pub fn new(
        net_config: NetworkConfig,
        stdp_config: StdpConfig,
        decoder_config: DecoderConfig,
        base_estimator: Box<dyn BaseEstimator>,
        burn_in_ticks: u64,
        seed: u64,
    ) -> Self {
        let encoder = Encoder::new();
        let network = SnnNetwork::new(encoder.input_dim, &net_config, seed);
        let stdp = StdpState::new(&network, stdp_config);
        let decoder = Decoder::new(&decoder_config);
        Self {
            network,
            stdp,
            encoder,
            decoder,
            base_estimator,
            net_config,
            tick_count: 0,
            burn_in_ticks,
            last_predicted_step_dt: None,
            last_tick_time: None,
            ratio_mean: 1.0,
            ratio_var: 0.0,
            confidence_alpha: config::DEFAULT_CONFIDENCE_ALPHA,
            confidence: 0.0,
            last_eta: None,
        }
    }

    pub fn tick(
        &mut self,
        position: u64,
        length: u64,
        elapsed: Duration,
        now: Instant,
    ) -> Option<Duration> {
        let elapsed_secs = elapsed.as_secs_f64();

        self.base_estimator.update(position, length, elapsed);
        let sps = self.base_estimator.steps_per_sec();

        let encoded = self.encoder.encode(elapsed_secs, position, length, sps);

        if encoded.features.len() != self.network.input_dim {
            self.network = SnnNetwork::new(
                encoded.features.len(), &self.net_config, 42,
            );
            self.stdp = StdpState::new(&self.network, self.stdp.config.clone());
        }

        self.stdp.begin_tick();
        let n = self.network.num_neurons;
        let neuron_config = crate::lif::NeuronConfig {
            v_threshold: self.net_config.v_threshold,
            v_reset: self.net_config.v_reset,
            refractory_steps: self.net_config.refractory_steps,
        };

        for neuron in &mut self.network.neurons {
            neuron.begin_tick();
        }

        let input_currents: Vec<f64> = self.network.input_synapses.iter().map(|syns| {
            syns.iter().map(|&(feat, w)| w * encoded.features[feat]).sum()
        }).collect();

        let temporal_neuron_count =
            (n as f64 * self.net_config.temporal_coding_frac) as usize;
        let temporal_spike_step =
            (encoded.temporal_spike_frac * self.net_config.steps_per_tick as f64) as u32;

        for step in 0..self.net_config.steps_per_tick {
            let mut recurrent_currents = vec![0.0; n];
            for (pre, syns) in self.network.recurrent_synapses.iter().enumerate() {
                if !self.network.neurons[pre].spiked {
                    continue;
                }
                for syn in syns {
                    recurrent_currents[syn.target] += syn.weight;
                }
            }

            for ni in 0..n {
                let mut current = input_currents[ni] + recurrent_currents[ni];
                if ni < temporal_neuron_count && step == temporal_spike_step {
                    current += self.net_config.v_threshold * 2.0;
                }
                self.network.neurons[ni].step(current, &neuron_config);
            }

            let spiked: Vec<bool> = self.network.neurons.iter().map(|n| n.spiked).collect();
            self.stdp.update_traces(
                step, &spiked, &self.network.recurrent_synapses,
                self.net_config.steps_per_tick,
            );
        }

        self.stdp.accumulate_eligibility();

        let output_idx = self.network.output_neuron();
        let output_rate = self.network.neurons[output_idx]
            .firing_rate(self.net_config.steps_per_tick);

        if let (Some(predicted_dt), Some(last_time)) =
            (self.last_predicted_step_dt, self.last_tick_time)
        {
            let actual_dt = now.duration_since(last_time).as_secs_f64();
            if predicted_dt > 0.0 && actual_dt > 0.0 {
                let ratio = actual_dt / predicted_dt;
                let signed_error = ((actual_dt - predicted_dt) / predicted_dt).clamp(-1.0, 1.0);

                let alpha = self.confidence_alpha;
                self.ratio_mean = alpha * ratio + (1.0 - alpha) * self.ratio_mean;
                let diff = ratio - self.ratio_mean;
                self.ratio_var = alpha * diff * diff + (1.0 - alpha) * self.ratio_var;
                self.confidence = 1.0 / (1.0 + self.ratio_var.sqrt());

                if self.tick_count > self.burn_in_ticks {
                    self.decoder.learn(ratio, output_rate);
                    self.stdp.apply_error_modulated_update(
                        signed_error,
                        &mut self.network.recurrent_synapses,
                        &self.network.is_excitatory,
                    );
                }
            }
        }

        let eta = self.base_estimator.estimate().map(|base_eta| {
            let raw_factor = self.decoder.decode(output_rate);
            let factor = self.confidence * raw_factor + (1.0 - self.confidence) * 1.0;
            let corrected_secs = base_eta.as_secs_f64() * factor;
            Duration::from_secs_f64(corrected_secs.max(0.0))
        });

        if let Some(ref eta_val) = eta {
            let remaining = length.saturating_sub(position);
            if remaining > 0 {
                self.last_predicted_step_dt = Some(eta_val.as_secs_f64() / remaining as f64);
            }
        }
        self.last_tick_time = Some(now);
        self.tick_count += 1;
        self.last_eta = eta;
        eta
    }

    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    pub fn reset(&mut self) {
        self.encoder.reset();
        let input_dim = self.encoder.input_dim;
        self.network = SnnNetwork::new(input_dim, &self.net_config, 42);
        self.stdp = StdpState::new(&self.network, self.stdp.config.clone());
        self.decoder.reset(&DecoderConfig::default());
        self.base_estimator.reset();
        self.tick_count = 0;
        self.last_predicted_step_dt = None;
        self.last_tick_time = None;
        self.ratio_mean = 1.0;
        self.ratio_var = 0.0;
        self.confidence = 0.0;
        self.last_eta = None;
    }
}

impl Clone for RsnnEtaCore {
    fn clone(&self) -> Self {
        Self {
            network: self.network.clone(),
            stdp: StdpState::new(&self.network, self.stdp.config.clone()),
            encoder: Encoder::new(),
            decoder: self.decoder.clone(),
            base_estimator: self.base_estimator.clone_box(),
            net_config: self.net_config.clone(),
            tick_count: self.tick_count,
            burn_in_ticks: self.burn_in_ticks,
            last_predicted_step_dt: self.last_predicted_step_dt,
            last_tick_time: None,
            ratio_mean: self.ratio_mean,
            ratio_var: self.ratio_var,
            confidence_alpha: self.confidence_alpha,
            confidence: self.confidence,
            last_eta: self.last_eta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimator::EmaEstimator;

    fn make_core() -> RsnnEtaCore {
        RsnnEtaCore::new(
            NetworkConfig::default(),
            StdpConfig::default(),
            DecoderConfig::default(),
            Box::new(EmaEstimator::default()),
            config::DEFAULT_BURN_IN_TICKS,
            42,
        )
    }

    #[test]
    fn test_new_core() {
        let core = make_core();
        assert!(core.last_eta.is_none());
        assert!((core.confidence() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_returns_none_before_warmup() {
        let mut core = make_core();
        let now = Instant::now();
        let eta = core.tick(1, 100, Duration::from_millis(100), now);
        assert!(eta.is_none(), "should return None before base estimator warms up");
    }

    #[test]
    fn test_returns_eta_after_warmup() {
        let mut core = RsnnEtaCore::new(
            NetworkConfig::default(),
            StdpConfig::default(),
            DecoderConfig::default(),
            Box::new(EmaEstimator::new(0.5, 2)),
            2,
            42,
        );
        let start = Instant::now();
        for i in 1..=10 {
            let elapsed = Duration::from_millis(i * 100);
            let now = start + elapsed;
            core.tick(i * 10, 1000, elapsed, now);
        }
        assert!(core.last_eta.is_some(), "should have ETA after warmup");
    }

    #[test]
    fn test_constant_rate_correction_near_unity() {
        let mut core = RsnnEtaCore::new(
            NetworkConfig::default(),
            StdpConfig::default(),
            DecoderConfig::default(),
            Box::new(EmaEstimator::new(0.5, 2)),
            2,
            42,
        );
        let start = Instant::now();
        for i in 1..=50 {
            let elapsed = Duration::from_millis(i * 100);
            let now = start + elapsed;
            core.tick(i * 10, 5000, elapsed, now);
        }
        if let Some(eta) = core.last_eta {
            let secs = eta.as_secs_f64();
            assert!(secs > 10.0 && secs < 100.0,
                "expected reasonable ETA, got {secs}s");
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut core = make_core();
        let now = Instant::now();
        core.tick(10, 100, Duration::from_secs(1), now);
        core.reset();
        assert!(core.last_eta.is_none());
        assert_eq!(core.tick_count, 0);
    }

    #[test]
    fn test_clone() {
        let core = make_core();
        let cloned = core.clone();
        assert_eq!(cloned.tick_count, core.tick_count);
        assert_eq!(cloned.burn_in_ticks, core.burn_in_ticks);
    }
}
