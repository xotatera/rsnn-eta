use std::fmt;
use std::path::PathBuf;
use std::sync::{mpsc, Mutex};
use std::time::Instant;

use indicatif::style::ProgressTracker;
use indicatif::ProgressState;

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
            signals_rx: Some(Mutex::new(rx)),
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

/// indicatif-compatible RSNN ETA estimator.
pub struct RsnnEta {
    pub core: RsnnEtaCore,
    pub signals_rx: Option<Mutex<mpsc::Receiver<Vec<f64>>>>,
    pub persistence_path: Option<PathBuf>,
}

impl RsnnEta {
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn builder() -> RsnnEtaBuilder {
        RsnnEtaBuilder::new()
    }

    pub fn save(&self) -> std::io::Result<()> {
        match &self.persistence_path {
            Some(path) => crate::persistence::save(&self.core, path),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "no persistence path configured",
            )),
        }
    }

    pub fn load(&mut self) -> std::io::Result<()> {
        match &self.persistence_path {
            Some(path) => {
                crate::persistence::load(&mut self.core, path)?;
                Ok(())
            }
            None => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "no persistence path configured",
            )),
        }
    }
}

impl Default for RsnnEta {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressTracker for RsnnEta {
    fn clone_box(&self) -> Box<dyn ProgressTracker> {
        Box::new(RsnnEta {
            core: self.core.clone(),
            signals_rx: None,
            persistence_path: self.persistence_path.clone(),
        })
    }

    fn tick(&mut self, state: &ProgressState, now: Instant) {
        if let Some(ref rx_mutex) = self.signals_rx {
            if let Ok(rx) = rx_mutex.lock() {
                while let Ok(signals) = rx.try_recv() {
                    self.core.encoder.set_side_channel(signals);
                }
            }
        }

        let position = state.pos();
        let length = state.len().unwrap_or(0);
        let elapsed = state.elapsed();

        self.core.tick(position, length, elapsed, now);
    }

    fn reset(&mut self, _state: &ProgressState, _now: Instant) {
        self.core.reset();
    }

    fn write(&self, _state: &ProgressState, w: &mut dyn fmt::Write) {
        match self.core.last_eta {
            Some(eta) => {
                let secs = eta.as_secs();
                if secs >= 3600 {
                    let _ = write!(w, "{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60);
                } else {
                    let _ = write!(w, "{:02}:{:02}", secs / 60, secs % 60);
                }
            }
            None => {
                let _ = write!(w, "--:--");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_default() {
        let eta = RsnnEta::new();
        assert!(eta.core.last_eta.is_none());
    }

    #[test]
    fn test_builder_defaults() {
        let eta = RsnnEta::builder().build();
        assert!(eta.core.last_eta.is_none());
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
}
