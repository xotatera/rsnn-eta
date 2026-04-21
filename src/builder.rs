use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::config::{self, DecoderConfig, NetworkConfig, StdpConfig};
use crate::estimator::{BaseEstimator, EmaEstimator};
use crate::tracker::RsnnEtaCore;

/// Builder for configuring an RSNN ETA estimator.
pub struct RsnnEtaBuilder {
    net_config: NetworkConfig,
    stdp_config: StdpConfig,
    decoder_config: DecoderConfig,
    base_estimator: Option<Box<dyn BaseEstimator>>,
    ema_alpha: f64,
    ema_warmup: u64,
    burn_in_ticks: u64,
    persistence_path: Option<PathBuf>,
    seed: u64,
}

impl Default for RsnnEtaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RsnnEtaBuilder {
    pub fn new() -> Self {
        Self {
            net_config: NetworkConfig::default(),
            stdp_config: StdpConfig::default(),
            decoder_config: DecoderConfig::default(),
            base_estimator: None,
            ema_alpha: config::DEFAULT_EMA_ALPHA,
            ema_warmup: config::DEFAULT_EMA_WARMUP,
            burn_in_ticks: config::DEFAULT_BURN_IN_TICKS,
            persistence_path: None,
            seed: 42,
        }
    }

    pub fn neurons(mut self, n: usize) -> Self {
        self.net_config.num_neurons = n;
        self
    }

    pub fn steps_per_tick(mut self, n: u32) -> Self {
        self.net_config.steps_per_tick = n;
        self
    }

    pub fn base_estimator(mut self, est: Box<dyn BaseEstimator>) -> Self {
        self.base_estimator = Some(est);
        self
    }

    pub fn ema_alpha(mut self, alpha: f64) -> Self {
        self.ema_alpha = alpha;
        self
    }

    pub fn burn_in_ticks(mut self, n: u64) -> Self {
        self.burn_in_ticks = n;
        self
    }

    pub fn persistence(mut self, path: impl Into<PathBuf>) -> Self {
        self.persistence_path = Some(path.into());
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    pub fn build(self) -> RsnnEta {
        let base = self.base_estimator.unwrap_or_else(|| {
            Box::new(EmaEstimator::new(self.ema_alpha, self.ema_warmup))
        });
        let core = RsnnEtaCore::new(
            self.net_config,
            self.stdp_config,
            self.decoder_config,
            base,
            self.burn_in_ticks,
            self.seed,
        );

        let mut eta = RsnnEta {
            core,
            signals_rx: None,
            persistence_path: self.persistence_path,
        };

        if let Some(ref path) = eta.persistence_path {
            if path.exists() {
                let _ = eta.load();
            }
        }

        eta
    }

    pub fn build_with_signals(self) -> (RsnnEta, mpsc::Sender<Vec<f64>>) {
        let (tx, rx) = mpsc::channel();
        let persistence_path = self.persistence_path.clone();

        let base = self.base_estimator.unwrap_or_else(|| {
            Box::new(EmaEstimator::new(self.ema_alpha, self.ema_warmup))
        });
        let core = RsnnEtaCore::new(
            self.net_config,
            self.stdp_config,
            self.decoder_config,
            base,
            self.burn_in_ticks,
            self.seed,
        );

        let mut eta = RsnnEta {
            core,
            signals_rx: Some(rx),
            persistence_path,
        };

        if let Some(ref path) = eta.persistence_path {
            if path.exists() {
                let _ = eta.load();
            }
        }

        (eta, tx)
    }
}

/// RSNN-based ETA estimator with STDP correction factor.
pub struct RsnnEta {
    pub core: RsnnEtaCore,
    pub signals_rx: Option<mpsc::Receiver<Vec<f64>>>,
    pub persistence_path: Option<PathBuf>,
}

impl RsnnEta {
    /// Zero-config constructor with defaults.
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Start building a customized estimator.
    pub fn builder() -> RsnnEtaBuilder {
        RsnnEtaBuilder::new()
    }

    /// Process one progress tick. Drains any pending side-channel signals,
    /// then delegates to the core. Returns corrected ETA or None if not warmed up.
    pub fn tick(
        &mut self,
        position: u64,
        length: u64,
        elapsed: Duration,
        now: Instant,
    ) -> Option<Duration> {
        // Drain side-channel signals
        if let Some(ref rx) = self.signals_rx {
            while let Ok(signals) = rx.try_recv() {
                self.core.encoder.set_side_channel(signals);
            }
        }

        self.core.tick(position, length, elapsed, now)
    }

    /// Reset all state (preserves loaded weights if persistence is enabled).
    pub fn reset(&mut self) {
        self.core.reset();
    }

    /// Get the last computed ETA.
    pub fn last_eta(&self) -> Option<Duration> {
        self.core.last_eta
    }

    /// Get current confidence level [0, 1].
    pub fn confidence(&self) -> f64 {
        self.core.confidence()
    }

    /// Save weights to the configured persistence path.
    pub fn save(&self) -> std::io::Result<()> {
        match &self.persistence_path {
            Some(path) => crate::persistence::save(&self.core, path),
            None => Err(std::io::Error::other("no persistence path configured")),
        }
    }

    /// Load weights from the configured persistence path.
    pub fn load(&mut self) -> std::io::Result<()> {
        match &self.persistence_path {
            Some(path) => {
                crate::persistence::load(&mut self.core, path)?;
                Ok(())
            }
            None => Err(std::io::Error::other("no persistence path configured")),
        }
    }
}

impl Default for RsnnEta {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_default() {
        let eta = RsnnEta::new();
        assert!(eta.last_eta().is_none());
    }

    #[test]
    fn test_builder_defaults() {
        let eta = RsnnEta::builder().build();
        assert!(eta.last_eta().is_none());
    }

    #[test]
    fn test_builder_custom_neurons() {
        let eta = RsnnEta::builder().neurons(100).build();
        assert_eq!(eta.core.network.num_neurons, 100);
    }

    #[test]
    fn test_builder_custom_steps() {
        let eta = RsnnEta::builder().steps_per_tick(30).build();
        assert_eq!(eta.core.net_config.steps_per_tick, 30);
    }

    #[test]
    fn test_builder_with_signals() {
        let (eta, tx) = RsnnEta::builder().build_with_signals();
        tx.send(vec![0.5, 0.8]).unwrap();
        assert!(eta.signals_rx.is_some());
    }

    #[test]
    fn test_builder_persistence_path() {
        let eta = RsnnEta::builder()
            .persistence("/tmp/test_weights.bin")
            .build();
        assert!(eta.persistence_path.is_some());
    }

    #[test]
    fn test_builder_seed() {
        let eta1 = RsnnEta::builder().seed(42).build();
        let eta2 = RsnnEta::builder().seed(42).build();
        assert_eq!(
            eta1.core.network.neurons[0].tau,
            eta2.core.network.neurons[0].tau,
        );
    }

    #[test]
    fn test_tick_and_eta() {
        let mut eta = RsnnEta::builder()
            .neurons(20)
            .steps_per_tick(5)
            .burn_in_ticks(2)
            .seed(42)
            .build();

        let start = Instant::now();
        for i in 1..=20 {
            let elapsed = Duration::from_millis(i * 100);
            eta.tick(i * 10, 1000, elapsed, start + elapsed);
        }
        assert!(eta.last_eta().is_some());
    }
}
