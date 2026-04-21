/// Encodes progress state into input features for the RSNN.
pub struct Encoder {
    elapsed_max: f64,
    delta_ema: f64,
    delta_ema_alpha: f64,
    last_tick_secs: Option<f64>,
    side_channel_dim: Option<usize>,
    side_channel_hold: Vec<f64>,
    pub input_dim: usize,
}

/// Encoded features from one tick.
pub struct EncodedInput {
    pub features: Vec<f64>,
    pub temporal_spike_frac: f64,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            elapsed_max: 1.0,
            delta_ema: 0.0,
            delta_ema_alpha: 0.1,
            last_tick_secs: None,
            side_channel_dim: None,
            side_channel_hold: Vec::new(),
            input_dim: 4,
        }
    }

    pub fn encode(
        &mut self,
        elapsed_secs: f64,
        position: u64,
        length: u64,
        steps_per_sec: f64,
    ) -> EncodedInput {
        let fraction = if length > 0 {
            position as f64 / length as f64
        } else {
            0.0
        };

        let rate_norm = 1.0 - (-steps_per_sec * 0.01).exp();

        if elapsed_secs > self.elapsed_max {
            self.elapsed_max = elapsed_secs;
        }
        let elapsed_norm = elapsed_secs / self.elapsed_max;

        let (delta_norm, temporal_spike_frac) = match self.last_tick_secs {
            Some(last) => {
                let delta = (elapsed_secs - last).max(0.0);
                if self.delta_ema <= 0.0 {
                    self.delta_ema = delta;
                } else {
                    self.delta_ema = self.delta_ema_alpha * delta
                        + (1.0 - self.delta_ema_alpha) * self.delta_ema;
                }
                let dn = if self.delta_ema > 0.0 {
                    (delta / self.delta_ema).min(2.0) / 2.0
                } else {
                    0.5
                };
                (dn, dn)
            }
            None => (0.5, 0.5),
        };
        self.last_tick_secs = Some(elapsed_secs);

        let mut features = vec![fraction, rate_norm, elapsed_norm, delta_norm];
        features.extend_from_slice(&self.side_channel_hold);

        EncodedInput { features, temporal_spike_frac }
    }

    pub fn set_side_channel(&mut self, signals: Vec<f64>) {
        match self.side_channel_dim {
            Some(dim) => {
                assert_eq!(signals.len(), dim, "side channel dimension mismatch");
            }
            None => {
                self.side_channel_dim = Some(signals.len());
                self.input_dim = 4 + signals.len();
            }
        }
        self.side_channel_hold = signals;
    }

    pub fn reset(&mut self) {
        self.elapsed_max = 1.0;
        self.delta_ema = 0.0;
        self.last_tick_secs = None;
        self.side_channel_dim = None;
        self.side_channel_hold.clear();
        self.input_dim = 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_encoder() {
        let enc = Encoder::new();
        assert_eq!(enc.input_dim, 4);
    }

    #[test]
    fn test_encode_basic() {
        let mut enc = Encoder::new();
        let out = enc.encode(1.0, 50, 100, 10.0);
        assert_eq!(out.features.len(), 4);
        assert!((out.features[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_encode_fraction_bounds() {
        let mut enc = Encoder::new();
        let out = enc.encode(0.0, 0, 100, 0.0);
        assert!((out.features[0]).abs() < 1e-10);
        let out = enc.encode(10.0, 100, 100, 10.0);
        assert!((out.features[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_encode_zero_length() {
        let mut enc = Encoder::new();
        let out = enc.encode(1.0, 0, 0, 0.0);
        assert!((out.features[0]).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_spike_frac() {
        let mut enc = Encoder::new();
        let out1 = enc.encode(1.0, 10, 100, 10.0);
        assert!((out1.temporal_spike_frac - 0.5).abs() < 1e-10);
        let out2 = enc.encode(1.01, 11, 100, 10.0);
        let out3 = enc.encode(2.0, 12, 100, 10.0);
        assert!(out3.temporal_spike_frac > out2.temporal_spike_frac);
    }

    #[test]
    fn test_side_channel() {
        let mut enc = Encoder::new();
        enc.set_side_channel(vec![0.5, 0.8]);
        assert_eq!(enc.input_dim, 6);
        let out = enc.encode(1.0, 50, 100, 10.0);
        assert_eq!(out.features.len(), 6);
        assert!((out.features[4] - 0.5).abs() < 1e-10);
        assert!((out.features[5] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_side_channel_zero_order_hold() {
        let mut enc = Encoder::new();
        enc.set_side_channel(vec![0.5]);
        let out1 = enc.encode(1.0, 10, 100, 10.0);
        assert!((out1.features[4] - 0.5).abs() < 1e-10);
        let out2 = enc.encode(2.0, 20, 100, 10.0);
        assert!((out2.features[4] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut enc = Encoder::new();
        enc.set_side_channel(vec![0.5]);
        enc.encode(1.0, 50, 100, 10.0);
        enc.reset();
        assert_eq!(enc.input_dim, 4);
        assert!(enc.last_tick_secs.is_none());
    }
}
