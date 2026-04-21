# rsnn-eta: RSNN+STDP Correction Factor ETA Estimator for indicatif

**Date:** 2026-04-21
**Status:** Approved
**Target:** Standalone Rust crate, indicatif-compatible via `ProgressTracker`
**Consumer:** [insights](~/Projects/insights/) — crypto trading system with long-running training, backtest, and evolution workloads

---

## Overview

`rsnn-eta` is a standalone Rust crate that provides an indicatif-compatible ETA estimator combining a pluggable base estimator with a Recurrent Spiking Neural Network (RSNN) correction layer trained via Spike-Timing-Dependent Plasticity (STDP).

The RSNN learns a correction factor applied on top of a base ETA estimate (default: EMA). This gives graceful cold-start behavior (correction ≈ 1.0 during burn-in, delegating to the base estimator) and lets the network focus on learning temporal pattern deviations — phase transitions, bursts, non-linear progress — rather than raw ETA prediction.

Architecturally aligned with the `rc-snn` crate in the insights project (LIF neurons, sparse E/I topology, vanilla STDP with eligibility traces) but reimplemented minimally for portability with zero dependency on the insights workspace.

---

## Architecture

```
Tick Events (position, elapsed, ...)
         │
         ▼
┌─────────────────────┐
│   Input Encoder     │  Rate coding: (fraction, rate, elapsed, delta_t)
│                     │  Temporal coding: inter-tick spike timing
│   + Optional side   │  User-injected signals via channel
│     channel signals │
└────────┬────────────┘
         │ spike trains
         ▼
┌─────────────────────┐
│   RSNN Reservoir    │  LIF neurons, sparse recurrent connectivity
│   (configurable     │  80/20 E/I split, log-uniform tau distribution
│    default ~50)     │  Vanilla STDP on recurrent synapses
└────────┬────────────┘
         │ firing rates
         ▼
┌─────────────────────┐
│   Correction        │  Output neuron firing rate → correction factor
│   Decoder           │  factor = exp(scale * (f - 0.5))
│                     │  Bounded below at epsilon, unbounded above
└────────┬────────────┘
         │ correction factor
         ▼
┌─────────────────────┐
│   Base Estimator    │  Pluggable trait, ships with EMA (default)
│   (trait object)    │  Computes raw ETA from tick history
└────────┬────────────┘
         │ base_eta
         ▼
    final_eta = base_eta × correction_factor

         │
         ▼
┌─────────────────────┐
│  Error Signal       │  Ratio: actual_dt / predicted_dt → adjusts decoder
│  (per tick)         │  Signed: (actual - predicted) normalized → modulates
│                     │  STDP eligibility traces
└─────────────────────┘
```

---

## Neuron Model: LIF

- Membrane dynamics: `dv/dt = -(v - v_rest) / tau + current`
- Configurable threshold (default 1.0), reset (default 0.0), refractory period (default 1 step)
- Tau drawn from log-uniform distribution, range [3, 120] (smaller than rc-snn's [3, 240] — ETA patterns operate on shorter timescales)
- Per-neuron spike tracking: `spiked` flag per step, `spike_count` per tick, `firing_rate = spike_count / steps_per_tick`

---

## Network Topology

**Default configuration (50 neurons):**

| Parameter | Default | Configurable |
|-----------|---------|-------------|
| Reservoir neurons | 50 | Yes |
| Steps per tick | 20 | Yes |
| Input connectivity | 30% sparse | Yes |
| Recurrent connectivity | 10% sparse | Yes |
| E/I ratio | 80% / 20% | Yes |
| Output neurons | 1 | No |

**Input layer:** 4 core features + N side-channel signals
- Core: fraction_complete, steps_per_sec, elapsed_secs (normalized), inter_tick_delta (normalized)
- Side-channel: appended dynamically, zero-order hold when no new values

**Reservoir:** Sparse recurrent connectivity with E/I sign constraints. No self-connections.

**Output:** Single neuron, firing rate decoded into correction factor.

**Weight initialization:**
- Input: Uniform[-1, 1] at sparsity %
- Recurrent: Uniform[0, init_scale] × E/I sign at sparsity %
- Soft bounds: `(w_max - |w|)^power` applied to all updates

---

## STDP & Learning

**Vanilla STDP on recurrent synapses:**
- LTP: `Δw = η * A+ * exp(-Δt / τ_stdp)` — post fires after pre
- LTD: `Δw = -η * A- * exp(-Δt / τ_stdp)` — pre fires after post
- Defaults: η=0.05, A+=0.10, A-=0.12 (LTD bias), τ_stdp=0.2×steps_per_tick
- Eligibility traces per synapse, accumulated with decay (default 0.95) across ticks
- Soft bounds prevent weight saturation

**Error-modulated weight update (per tick):**

1. **Ratio signal:** `r = actual_step_dt / predicted_step_dt`
   - Drives decoder scale adjustment via gradient on `(r - 1.0)`

2. **Signed error:** `e = clamp((actual - predicted) / predicted, -1, 1)`
   - Modulates STDP eligibility: `Δw_final = eligibility[syn] * e * η_error`
   - Positive error (underestimate) → reinforce patterns increasing correction
   - Negative error (overestimate) → reinforce patterns decreasing correction

**Burn-in:** No learning during first N ticks (default 10). Network runs but weights frozen, correction factor decays toward 1.0.

**Confidence metric:** Running variance of ratio signal. High variance → correction factor damped toward 1.0. Low variance → RSNN gets more influence.
- `final_factor = confidence * factor + (1 - confidence) * 1.0`

---

## Input Encoding

**Rate coding (all ticks):**
- `fraction_complete`: position / length ∈ [0, 1]
- `steps_per_sec`: instantaneous rate from base estimator
- `elapsed_secs`: normalized by running max to ∈ [0, 1]
- `inter_tick_delta`: time since last tick, normalized by running EMA of deltas

Features multiplied through sparse input synapses as constant current for the simulation step duration.

**Temporal coding (per tick):**
- Inter-tick interval encoded as spike time at simulation start: shorter interval → earlier spike, longer → later
- Injected into first 20% of reservoir neurons
- Preserves raw timing information that rate coding smooths away

**Side-channel signals:**
- Received via `mpsc::Sender<Vec<f64>>` from `RsnnEta::with_signals()`
- Appended to input feature vector
- Zero-order hold when no new values received

---

## Output Decoding

- Single output neuron firing rate `f ∈ [0, 1]`
- Correction factor: `factor = exp(scale * (f - 0.5))`
  - `f = 0.5` → factor = 1.0 (no correction)
  - `f > 0.5` → factor > 1.0 (base ETA too optimistic)
  - `f < 0.5` → factor < 1.0 (base ETA too pessimistic)
- `scale` is learnable, initialized to 2.0, adjusted by ratio error signal
- Bounded below at epsilon (0.01) to prevent zero/negative ETA
- Confidence blending applied before final output

---

## Base Estimator Trait

```rust
pub trait BaseEstimator: Send + Sync {
    fn update(&mut self, position: u64, length: u64, elapsed: Duration);
    fn estimate(&self) -> Option<Duration>;
    fn is_warm(&self) -> bool;
    fn reset(&mut self);
}
```

**Built-in EMA estimator:**
- Tracks `ema_step_duration` with configurable alpha (default 0.05)
- Warm after configurable tick count (default 10)
- `estimate()` returns `remaining_steps * ema_step_duration`
- Handles zero-position and zero-length (returns None)

Users implement `BaseEstimator` for custom baselines. Builder accepts `Box<dyn BaseEstimator>`.

---

## Public API

**Zero-config:**
```rust
let style = ProgressStyle::with_template("{bar} {pos}/{len} ETA: {rsnn_eta}")
    .unwrap()
    .with_key("rsnn_eta", RsnnEta::new());
```

**Builder:**
```rust
let (eta, signals_tx) = RsnnEta::builder()
    .neurons(100)
    .steps_per_tick(30)
    .base_estimator(Box::new(MyCustomEstimator::new()))
    .ema_alpha(0.03)
    .burn_in_ticks(15)
    .persistence("./eta_weights.bin")
    .build_with_signals();

let eta = RsnnEta::builder().neurons(80).build();
```

**ProgressTracker impl:**
- `tick()`: base estimator update, RSNN simulation, STDP, error signal, confidence update
- `reset()`: reinitialize state (preserves loaded weights if persistence enabled)
- `write()`: formats `final_eta` as `HumanDuration`
- `clone_box()`: full state clone

**Persistence:**
- `save(&self, path: &Path) -> io::Result<()>`
- `load(path: &Path) -> io::Result<Self>`
- Serde + bincode, compact binary
- Auto-loads on construction if persistence path set and file exists

**Side-channel:**
```rust
signals_tx.send(vec![0.8, 1.2]).unwrap();
```

---

## Crate Structure

```
src/
├── lib.rs              # Public API, re-exports
├── lif.rs              # LIF neuron model
├── network.rs          # RSNN reservoir (topology, simulation step)
├── stdp.rs             # STDP traces, eligibility, weight updates
├── encoder.rs          # Rate + temporal input encoding
├── decoder.rs          # Output neuron → correction factor
├── estimator.rs        # BaseEstimator trait + EMA implementation
├── tracker.rs          # ProgressTracker impl, tick/error/confidence logic
├── builder.rs          # Builder pattern API
├── persistence.rs      # Save/load weights (serde + bincode)
└── config.rs           # Default constants, NetworkConfig, StdpConfig
```

**Dependencies:**
- `indicatif` — `ProgressTracker`, `ProgressState`, formatting
- `serde` + `bincode` — persistence
- `rand` — initialization
- No nalgebra (plain `Vec<Vec<f64>>`)
- No async runtime

---

## Testing & Performance

**Unit tests:**
- LIF: threshold firing, refractory, tau decay
- STDP: LTP/LTD windows, soft bounds, eligibility decay
- Encoder: rate normalization, temporal spike timing
- Decoder: f=0.5→1.0, symmetry, bounds
- EMA: warmup, convergence
- Persistence: round-trip save/load

**Integration tests:**
- Constant rate → correction converges to ~1.0
- Phase transition → correction adapts, ETA improves vs raw EMA
- Cold start → base estimator's ETA unmodified for burn-in ticks
- Side channel → changes network behavior
- Thread safety → `Arc<ProgressBar>` with `ProgressTracker`

**Performance targets:**
- Inference (50 neurons, 20 steps): < 50μs per tick
- Memory (50 neurons, 10% sparsity): < 100KB
- Criterion benchmarks for regression detection
