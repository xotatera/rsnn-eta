//! # rsnn-eta
//!
//! A biologically-inspired ETA estimator using a Recurrent Spiking Neural Network (RSNN)
//! with Spike-Timing-Dependent Plasticity (STDP) to learn correction factors for
//! time-remaining predictions.
//!
//! The core idea: a pluggable base estimator (default: exponential moving average) provides
//! a naive ETA, and an RSNN learns a correction factor from prediction errors. The network
//! adapts online via STDP, detecting phase transitions, bursts, and non-linear progress
//! patterns that defeat simple smoothing.
//!
//! ## Quick start
//!
//! ```rust
//! use std::time::{Duration, Instant};
//! use rsnn_eta::RsnnEta;
//!
//! let mut eta = RsnnEta::new();
//! let start = Instant::now();
//!
//! for i in 1..=100 {
//!     let elapsed = Duration::from_millis(i * 50);
//!     if let Some(remaining) = eta.tick(i * 10, 10_000, elapsed, start + elapsed) {
//!         // `remaining` is the corrected ETA as a Duration
//!     }
//! }
//! ```
//!
//! ## Builder
//!
//! ```rust
//! use rsnn_eta::RsnnEta;
//!
//! let mut eta = RsnnEta::builder()
//!     .neurons(100)          // reservoir size (default: 50)
//!     .steps_per_tick(30)    // simulation steps per tick (default: 20)
//!     .burn_in_ticks(15)     // ticks before STDP learning starts (default: 10)
//!     .ema_alpha(0.03)       // base EMA smoothing (default: 0.05)
//!     .seed(123)             // RNG seed for reproducibility
//!     .persistence("./weights.bin")  // optional save/load path
//!     .build();
//! ```
//!
//! ## Custom base estimator
//!
//! ```rust
//! use std::time::Duration;
//! use rsnn_eta::BaseEstimator;
//!
//! struct MyEstimator { /* ... */ }
//!
//! impl BaseEstimator for MyEstimator {
//!     fn update(&mut self, position: u64, length: u64, elapsed: Duration) { /* ... */ }
//!     fn estimate(&self) -> Option<Duration> { None }
//!     fn is_warm(&self) -> bool { false }
//!     fn reset(&mut self) {}
//!     fn steps_per_sec(&self) -> f64 { 0.0 }
//!     fn clone_box(&self) -> Box<dyn BaseEstimator> { Box::new(MyEstimator {}) }
//! }
//!
//! let eta = rsnn_eta::RsnnEta::builder()
//!     .base_estimator(Box::new(MyEstimator {}))
//!     .build();
//! ```
//!
//! ## Side-channel signals
//!
//! Inject additional features (e.g., batch size, phase indicator) beyond what the
//! progress bar exposes:
//!
//! ```rust
//! let (mut eta, tx) = rsnn_eta::RsnnEta::builder().build_with_signals();
//! tx.send(vec![0.8, 1.2]).unwrap(); // e.g., batch_size_ratio, phase_indicator
//! ```
//!
//! ## Architecture
//!
//! ```text
//! tick(pos, len, elapsed, now)
//!   │
//!   ├─► Base Estimator (EMA) ──► base_eta
//!   │
//!   ├─► Encoder (rate + temporal coding)
//!   │     │
//!   │     ▼
//!   │   RSNN Reservoir (LIF neurons, sparse E/I, STDP)
//!   │     │
//!   │     ▼
//!   │   Decoder ──► correction_factor
//!   │
//!   └─► final_eta = base_eta × (confidence × factor + (1-confidence) × 1.0)
//! ```

pub mod config;
pub mod lif;
pub mod network;
pub mod stdp;
pub mod encoder;
pub mod decoder;
pub mod estimator;
pub mod tracker;
pub mod builder;
pub mod persistence;

pub use builder::{RsnnEta, RsnnEtaBuilder};
pub use estimator::{BaseEstimator, EmaEstimator};
pub use config::{NetworkConfig, StdpConfig, DecoderConfig};
