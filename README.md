# rsnn-eta

A biologically-inspired ETA estimator using a Recurrent Spiking Neural Network (RSNN) with Spike-Timing-Dependent Plasticity (STDP).

## What it does

Predicts time remaining for long-running tasks. A pluggable base estimator (default: EMA) provides a naive ETA, and an RSNN learns a correction factor from prediction errors. The network adapts online via STDP, detecting phase transitions, bursts, and non-linear progress patterns that defeat simple smoothing.

## Architecture

```
tick(pos, len, elapsed, now)
  |
  +-> Base Estimator (EMA) --> base_eta
  |
  +-> Encoder (rate + temporal coding)
  |     |
  |     v
  |   RSNN Reservoir (LIF neurons, sparse E/I, STDP)
  |     |
  |     v
  |   Decoder --> correction_factor
  |
  +-> final_eta = base_eta * (confidence * factor + (1-confidence) * 1.0)
```

- **LIF neurons** with log-uniform time constants and configurable E/I ratio
- **Vanilla STDP** with error-modulated eligibility traces
- **Confidence blending** — correction factor is damped toward 1.0 when predictions are noisy
- **Burn-in period** — weights are frozen until the base estimator warms up

## Usage

```rust
use std::time::{Duration, Instant};
use rsnn_eta::RsnnEta;

let mut eta = RsnnEta::new();
let start = Instant::now();

for i in 1..=100 {
    let elapsed = Duration::from_millis(i * 50);
    if let Some(remaining) = eta.tick(i * 10, 10_000, elapsed, start + elapsed) {
        println!("ETA: {remaining:?}");
    }
}
```

### Builder

```rust
use rsnn_eta::RsnnEta;

let mut eta = RsnnEta::builder()
    .neurons(100)          // reservoir size (default: 50)
    .steps_per_tick(30)    // LIF simulation steps per tick (default: 20)
    .burn_in_ticks(15)     // ticks before STDP learning starts (default: 10)
    .ema_alpha(0.03)       // base EMA smoothing factor (default: 0.05)
    .seed(123)             // RNG seed for reproducibility
    .persistence("./weights.bin")  // optional weight save/load
    .build();
```

### Custom base estimator

```rust
use std::time::Duration;
use rsnn_eta::BaseEstimator;

struct MyEstimator { /* ... */ }

impl BaseEstimator for MyEstimator {
    fn update(&mut self, position: u64, length: u64, elapsed: Duration) { /* ... */ }
    fn estimate(&self) -> Option<Duration> { todo!() }
    fn is_warm(&self) -> bool { todo!() }
    fn reset(&mut self) {}
    fn steps_per_sec(&self) -> f64 { 0.0 }
    fn clone_box(&self) -> Box<dyn BaseEstimator> { todo!() }
}

let eta = rsnn_eta::RsnnEta::builder()
    .base_estimator(Box::new(MyEstimator {}))
    .build();
```

### Side-channel signals

Inject additional features beyond standard progress state:

```rust
let (mut eta, tx) = rsnn_eta::RsnnEta::builder().build_with_signals();

// From your workload code:
tx.send(vec![0.8, 1.2]).unwrap(); // e.g., batch_size_ratio, phase_indicator
```

The dimension is fixed on first send and held via zero-order hold between updates.

### Weight persistence

```rust
let mut eta = rsnn_eta::RsnnEta::builder()
    .persistence("./eta_weights.bin")
    .build();  // auto-loads if file exists

// ... run workload ...

eta.save().unwrap();  // persist learned weights for next run
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `neurons` | 50 | Reservoir neuron count |
| `steps_per_tick` | 20 | LIF simulation steps per progress tick |
| `burn_in_ticks` | 10 | Ticks before STDP learning activates |
| `ema_alpha` | 0.05 | EMA smoothing for default base estimator |
| `seed` | 42 | RNG seed for network initialization |

Advanced configuration via `NetworkConfig`, `StdpConfig`, and `DecoderConfig` structs.

## License

MIT
