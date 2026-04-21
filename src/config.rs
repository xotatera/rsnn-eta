use serde::{Deserialize, Serialize};

// ── Network defaults ──
pub const DEFAULT_NUM_NEURONS: usize = 50;
pub const DEFAULT_STEPS_PER_TICK: u32 = 20;
pub const DEFAULT_INPUT_SPARSITY: f64 = 0.3;
pub const DEFAULT_RECURRENT_SPARSITY: f64 = 0.1;
pub const DEFAULT_EXCITATORY_FRAC: f64 = 0.8;
pub const DEFAULT_INIT_INPUT_SCALE: f64 = 1.0;
pub const DEFAULT_INIT_RECURRENT_SCALE: f64 = 1.0;
pub const DEFAULT_TEMPORAL_CODING_FRAC: f64 = 0.2;

// ── Neuron defaults ──
pub const DEFAULT_V_THRESHOLD: f64 = 1.0;
pub const DEFAULT_V_RESET: f64 = 0.0;
pub const DEFAULT_REFRACTORY_STEPS: u8 = 1;
pub const DEFAULT_TAU_MIN: f64 = 3.0;
pub const DEFAULT_TAU_MAX: f64 = 120.0;

// ── STDP defaults ──
pub const DEFAULT_ETA_STDP: f64 = 0.05;
pub const DEFAULT_A_PLUS: f64 = 0.10;
pub const DEFAULT_A_MINUS: f64 = 0.12;
pub const DEFAULT_W_MAX: f64 = 1.0;
pub const DEFAULT_W_MIN: f64 = -1.0;
pub const DEFAULT_ELIGIBILITY_DECAY: f64 = 0.95;
pub const DEFAULT_TAU_STDP_FRAC: f64 = 0.2;
pub const DEFAULT_SOFT_BOUND_POWER: f64 = 1.0;
pub const DEFAULT_ETA_ERROR: f64 = 0.01;

// ── Decoder defaults ──
pub const DEFAULT_DECODER_SCALE: f64 = 2.0;
pub const DEFAULT_DECODER_EPSILON: f64 = 0.01;
pub const DEFAULT_DECODER_SCALE_LR: f64 = 0.01;

// ── Tracker defaults ──
pub const DEFAULT_BURN_IN_TICKS: u64 = 10;
pub const DEFAULT_CONFIDENCE_ALPHA: f64 = 0.1;

// ── EMA defaults ──
pub const DEFAULT_EMA_ALPHA: f64 = 0.05;
pub const DEFAULT_EMA_WARMUP: u64 = 10;

/// Network topology configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub num_neurons: usize,
    pub steps_per_tick: u32,
    pub input_sparsity: f64,
    pub recurrent_sparsity: f64,
    pub excitatory_frac: f64,
    pub init_input_scale: f64,
    pub init_recurrent_scale: f64,
    pub tau_min: f64,
    pub tau_max: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub refractory_steps: u8,
    pub temporal_coding_frac: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            num_neurons: DEFAULT_NUM_NEURONS,
            steps_per_tick: DEFAULT_STEPS_PER_TICK,
            input_sparsity: DEFAULT_INPUT_SPARSITY,
            recurrent_sparsity: DEFAULT_RECURRENT_SPARSITY,
            excitatory_frac: DEFAULT_EXCITATORY_FRAC,
            init_input_scale: DEFAULT_INIT_INPUT_SCALE,
            init_recurrent_scale: DEFAULT_INIT_RECURRENT_SCALE,
            tau_min: DEFAULT_TAU_MIN,
            tau_max: DEFAULT_TAU_MAX,
            v_threshold: DEFAULT_V_THRESHOLD,
            v_reset: DEFAULT_V_RESET,
            refractory_steps: DEFAULT_REFRACTORY_STEPS,
            temporal_coding_frac: DEFAULT_TEMPORAL_CODING_FRAC,
        }
    }
}

/// STDP learning configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StdpConfig {
    pub eta_stdp: f64,
    pub a_plus: f64,
    pub a_minus: f64,
    pub w_max: f64,
    pub w_min: f64,
    pub eligibility_decay: f64,
    pub tau_stdp_frac: f64,
    pub soft_bound_power: f64,
    pub eta_error: f64,
}

impl StdpConfig {
    pub fn tau_stdp(&self, steps_per_tick: u32) -> f64 {
        (steps_per_tick as f64 * self.tau_stdp_frac).max(2.0)
    }

    pub fn soft_bound(&self, w: f64) -> f64 {
        if w >= 0.0 {
            (self.w_max - w).max(0.0).powf(self.soft_bound_power)
        } else {
            (w - self.w_min).abs().max(0.0).powf(self.soft_bound_power)
        }
    }
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            eta_stdp: DEFAULT_ETA_STDP,
            a_plus: DEFAULT_A_PLUS,
            a_minus: DEFAULT_A_MINUS,
            w_max: DEFAULT_W_MAX,
            w_min: DEFAULT_W_MIN,
            eligibility_decay: DEFAULT_ELIGIBILITY_DECAY,
            tau_stdp_frac: DEFAULT_TAU_STDP_FRAC,
            soft_bound_power: DEFAULT_SOFT_BOUND_POWER,
            eta_error: DEFAULT_ETA_ERROR,
        }
    }
}

/// Decoder configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecoderConfig {
    pub initial_scale: f64,
    pub epsilon: f64,
    pub scale_lr: f64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            initial_scale: DEFAULT_DECODER_SCALE,
            epsilon: DEFAULT_DECODER_EPSILON,
            scale_lr: DEFAULT_DECODER_SCALE_LR,
        }
    }
}
