use crate::config::StdpConfig;
use crate::network::{SnnNetwork, Synapse};

/// STDP learning state.
#[derive(Clone)]
pub struct StdpState {
    pub last_spike: Vec<i32>,
    pub traces: Vec<Vec<f64>>,
    pub eligibility: Vec<Vec<f64>>,
    pub config: StdpConfig,
}

impl StdpState {
    pub fn new(network: &SnnNetwork, config: StdpConfig) -> Self {
        let n = network.num_neurons;
        let traces: Vec<Vec<f64>> = network.recurrent_synapses.iter()
            .map(|syns| vec![0.0; syns.len()])
            .collect();
        let eligibility: Vec<Vec<f64>> = network.recurrent_synapses.iter()
            .map(|syns| vec![0.0; syns.len()])
            .collect();
        Self {
            last_spike: vec![-1; n],
            traces,
            eligibility,
            config,
        }
    }

    pub fn update_traces(
        &mut self,
        step: u32,
        neurons_spiked: &[bool],
        recurrent_synapses: &[Vec<Synapse>],
        steps_per_tick: u32,
    ) {
        let tau = self.config.tau_stdp(steps_per_tick);
        let step_i = step as i32;

        for (pre, syns) in recurrent_synapses.iter().enumerate() {
            for (si, syn) in syns.iter().enumerate() {
                let post = syn.target;

                // LTP: pre fired earlier, post fires now
                if neurons_spiked[post] && self.last_spike[pre] >= 0 {
                    let dt = (step_i - self.last_spike[pre]) as f64;
                    if dt > 0.0 {
                        self.traces[pre][si] += self.config.a_plus * (-dt / tau).exp();
                    }
                }

                // LTD: post fired earlier, pre fires now
                if neurons_spiked[pre] && self.last_spike[post] >= 0 {
                    let dt = (step_i - self.last_spike[post]) as f64;
                    if dt > 0.0 {
                        self.traces[pre][si] -= self.config.a_minus * (-dt / tau).exp();
                    }
                }
            }
        }

        for (i, &spiked) in neurons_spiked.iter().enumerate() {
            if spiked {
                self.last_spike[i] = step_i;
            }
        }
    }

    pub fn accumulate_eligibility(&mut self) {
        let decay = self.config.eligibility_decay;
        for (pre, traces) in self.traces.iter().enumerate() {
            for (si, &trace) in traces.iter().enumerate() {
                self.eligibility[pre][si] = self.eligibility[pre][si] * decay + trace;
            }
        }
    }

    pub fn apply_error_modulated_update(
        &self,
        error: f64,
        recurrent_synapses: &mut [Vec<Synapse>],
        is_excitatory: &[bool],
    ) {
        let eta = self.config.eta_error;
        for (pre, syns) in recurrent_synapses.iter_mut().enumerate() {
            let (lo, hi) = if is_excitatory[pre] {
                (0.0, self.config.w_max)
            } else {
                (self.config.w_min, 0.0)
            };
            for (si, syn) in syns.iter_mut().enumerate() {
                let elig = self.eligibility[pre][si];
                if elig.abs() < 1e-12 {
                    continue;
                }
                let sb = self.config.soft_bound(syn.weight);
                let dw = eta * elig * error * sb;
                syn.weight = (syn.weight + dw).clamp(lo, hi);
            }
        }
    }

    pub fn begin_tick(&mut self) {
        self.last_spike.fill(-1);
        for traces in &mut self.traces {
            traces.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NetworkConfig;

    fn make_test_network() -> SnnNetwork {
        SnnNetwork::new(4, &NetworkConfig::default(), 42)
    }

    #[test]
    fn test_new_state_dimensions() {
        let net = make_test_network();
        let state = StdpState::new(&net, StdpConfig::default());
        assert_eq!(state.last_spike.len(), net.num_neurons);
        assert_eq!(state.traces.len(), net.num_neurons);
        assert_eq!(state.eligibility.len(), net.num_neurons);
        for (pre, syns) in net.recurrent_synapses.iter().enumerate() {
            assert_eq!(state.traces[pre].len(), syns.len());
            assert_eq!(state.eligibility[pre].len(), syns.len());
        }
    }

    #[test]
    fn test_begin_tick_resets() {
        let net = make_test_network();
        let mut state = StdpState::new(&net, StdpConfig::default());
        state.last_spike[0] = 5;
        if !state.traces.is_empty() && !state.traces[0].is_empty() {
            state.traces[0][0] = 0.5;
        }
        state.begin_tick();
        assert!(state.last_spike.iter().all(|&s| s == -1));
        assert!(state.traces.iter().all(|t| t.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn test_ltp_pre_before_post() {
        let net = make_test_network();
        let cfg = StdpConfig::default();
        let steps_per_tick = 20u32;
        let (pre, syn_idx, post) = find_synapse(&net);
        let mut state = StdpState::new(&net, cfg);
        state.begin_tick();
        let mut spiked = vec![false; net.num_neurons];
        spiked[pre] = true;
        state.update_traces(2, &spiked, &net.recurrent_synapses, steps_per_tick);
        spiked.fill(false);
        spiked[post] = true;
        state.update_traces(5, &spiked, &net.recurrent_synapses, steps_per_tick);
        assert!(state.traces[pre][syn_idx] > 0.0, "expected positive trace (LTP)");
    }

    #[test]
    fn test_ltd_post_before_pre() {
        let net = make_test_network();
        let cfg = StdpConfig::default();
        let steps_per_tick = 20u32;
        let (pre, syn_idx, post) = find_synapse(&net);
        let mut state = StdpState::new(&net, cfg);
        state.begin_tick();
        let mut spiked = vec![false; net.num_neurons];
        spiked[post] = true;
        state.update_traces(2, &spiked, &net.recurrent_synapses, steps_per_tick);
        spiked.fill(false);
        spiked[pre] = true;
        state.update_traces(5, &spiked, &net.recurrent_synapses, steps_per_tick);
        assert!(state.traces[pre][syn_idx] < 0.0, "expected negative trace (LTD)");
    }

    #[test]
    fn test_eligibility_accumulation_and_decay() {
        let net = make_test_network();
        let cfg = StdpConfig::default();
        let mut state = StdpState::new(&net, cfg.clone());
        if !state.traces.is_empty() && !state.traces[0].is_empty() {
            state.traces[0][0] = 1.0;
        }
        state.accumulate_eligibility();
        if !state.eligibility.is_empty() && !state.eligibility[0].is_empty() {
            let e1 = state.eligibility[0][0];
            assert!((e1 - 1.0).abs() < 1e-10, "first accumulation should be 1.0");
            state.traces[0][0] = 0.0;
            state.accumulate_eligibility();
            let e2 = state.eligibility[0][0];
            assert!((e2 - cfg.eligibility_decay).abs() < 1e-10,
                "should decay to {}, got {e2}", cfg.eligibility_decay);
        }
    }

    #[test]
    fn test_error_modulated_update_respects_sign() {
        let mut net = make_test_network();
        let cfg = StdpConfig::default();
        let mut state = StdpState::new(&net, cfg);
        if !state.eligibility.is_empty() && !state.eligibility[0].is_empty() {
            state.eligibility[0][0] = 1.0;
            let w_before = net.recurrent_synapses[0][0].weight;
            let is_exc = net.is_excitatory[0];
            state.apply_error_modulated_update(1.0, &mut net.recurrent_synapses, &net.is_excitatory);
            let w_after = net.recurrent_synapses[0][0].weight;
            if is_exc {
                assert!(w_after >= w_before, "excitatory weight should increase");
            } else {
                assert!(w_after <= w_before, "inhibitory weight should decrease (more negative)");
            }
        }
    }

    #[test]
    fn test_soft_bounds_prevent_saturation() {
        let cfg = StdpConfig::default();
        let sb_near_max = cfg.soft_bound(cfg.w_max - 0.01);
        let sb_near_zero = cfg.soft_bound(0.01);
        assert!(sb_near_max < sb_near_zero, "soft bound should be smaller near w_max");
    }

    fn find_synapse(net: &SnnNetwork) -> (usize, usize, usize) {
        for (pre, syns) in net.recurrent_synapses.iter().enumerate() {
            if let Some(syn) = syns.first() {
                return (pre, 0, syn.target);
            }
        }
        panic!("no synapses found in network");
    }
}
