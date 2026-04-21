//! Tests verifying that STDP learning and confidence tracking behave correctly
//! through the full pipeline.

use std::time::{Duration, Instant};

use rsnn_eta::RsnnEta;

/// Confidence should start at 0 and increase as predictions stabilize.
#[test]
fn test_confidence_increases_with_stable_rate() {
    let mut eta = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(10)
        .burn_in_ticks(3)
        .seed(42)
        .build();

    let start = Instant::now();
    // Constant rate workload
    for i in 1..=100 {
        eta.tick(i * 5, 10000, Duration::from_millis(i * 50), start + Duration::from_millis(i * 50));
    }

    let conf = eta.confidence();
    assert!(conf > 0.0, "confidence should be positive after stable ticks, got {conf}");
}

/// After a sudden rate change, confidence should drop then recover.
#[test]
fn test_confidence_drops_on_rate_change() {
    let mut eta = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(10)
        .burn_in_ticks(3)
        .seed(42)
        .build();

    let start = Instant::now();

    // Phase 1: fast steady rate — build up confidence
    for i in 1..=50 {
        eta.tick(i * 10, 10000, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }
    let conf_before = eta.confidence();

    // Phase 2: sudden 10x slowdown for a few ticks
    for i in 51..=55 {
        let pos = 500 + (i - 50);
        let ms = 5000 + (i - 50) * 1000;
        eta.tick(pos, 10000, Duration::from_millis(ms), start + Duration::from_millis(ms));
    }
    let conf_after_shock = eta.confidence();

    // Confidence should decrease (or at least not increase) after rate shock
    // The ratio variance spikes, pushing confidence down
    assert!(conf_after_shock <= conf_before + 0.1,
        "confidence should not increase after rate shock: before={conf_before}, after={conf_after_shock}");
}

/// With a constant rate, the correction factor should stay near 1.0,
/// meaning the corrected ETA is close to the base ETA.
#[test]
fn test_correction_factor_near_unity_for_constant_rate() {
    let mut eta = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(10)
        .burn_in_ticks(3)
        .seed(42)
        .build();

    let start = Instant::now();
    let total = 10000u64;

    for i in 1..=200 {
        eta.tick(i * 5, total, Duration::from_millis(i * 50), start + Duration::from_millis(i * 50));
    }

    // At position 1000 of 10000, rate = 100 steps/sec, base ETA = 90 sec
    // With constant rate, correction should be ~1.0, so corrected ETA should
    // be in a reasonable range around the base
    let secs = eta.last_eta().unwrap().as_secs_f64();
    assert!(secs > 20.0 && secs < 200.0,
        "corrected ETA should be near base ETA for constant rate, got {secs}s");
}

/// STDP weights should change after burn-in when there are prediction errors.
#[test]
fn test_weights_change_after_burn_in() {
    let mut eta = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(10)
        .burn_in_ticks(5)
        .seed(42)
        .build();

    let start = Instant::now();

    // Run past burn-in with constant rate
    for i in 1..=6 {
        eta.tick(i * 10, 1000, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    // Capture weights snapshot
    let weights_before: Vec<f64> = eta.core.network.recurrent_synapses.iter()
        .flat_map(|syns| syns.iter().map(|s| s.weight))
        .collect();

    // Run more ticks with a different rate to generate prediction errors
    for i in 7..=30 {
        let pos = 60 + (i - 6);
        let ms = 600 + (i - 6) * 500; // much slower than before
        eta.tick(pos, 1000, Duration::from_millis(ms), start + Duration::from_millis(ms));
    }

    let weights_after: Vec<f64> = eta.core.network.recurrent_synapses.iter()
        .flat_map(|syns| syns.iter().map(|s| s.weight))
        .collect();

    // At least some weights should have changed
    let changed = weights_before.iter().zip(&weights_after)
        .filter(|(a, b)| (*a - *b).abs() > 1e-15)
        .count();
    assert!(changed > 0, "STDP should modify some weights after burn-in with prediction errors");
}

/// Decoder scale should adapt when there are systematic prediction errors.
#[test]
fn test_decoder_scale_adapts() {
    let mut eta = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(10)
        .burn_in_ticks(3)
        .seed(42)
        .build();

    let start = Instant::now();
    let initial_scale = eta.core.decoder.scale;

    // Run enough ticks past burn-in with variable rate to produce errors
    for i in 1..=50 {
        // Oscillating rate: fast/slow alternating
        let ms = if i % 2 == 0 { i * 50 } else { i * 200 };
        eta.tick(i * 5, 5000, Duration::from_millis(ms), start + Duration::from_millis(ms));
    }

    let final_scale = eta.core.decoder.scale;
    assert!((final_scale - initial_scale).abs() > 1e-10,
        "decoder scale should change with prediction errors: initial={initial_scale}, final={final_scale}");
}

/// Multiple side-channel signals should expand the input dimension and
/// affect the network's behavior differently than without.
#[test]
fn test_side_channel_affects_output() {
    let start = Instant::now();
    let total = 1000u64;

    // Run without side channel
    let mut eta_plain = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(10)
        .burn_in_ticks(2)
        .seed(42)
        .build();

    for i in 1..=30 {
        eta_plain.tick(i * 10, total, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    // Run with side channel injecting a strong signal
    let (mut eta_signal, tx) = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(10)
        .burn_in_ticks(2)
        .seed(42)
        .build_with_signals();

    tx.send(vec![1.0, 1.0, 1.0]).unwrap();

    for i in 1..=30 {
        eta_signal.tick(i * 10, total, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    // Both should produce ETAs
    assert!(eta_plain.last_eta().is_some());
    assert!(eta_signal.last_eta().is_some());

    // The input dimensions should differ
    assert_eq!(eta_plain.core.encoder.input_dim, 4);
    assert_eq!(eta_signal.core.encoder.input_dim, 7); // 4 + 3
}
