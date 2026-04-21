//! Edge case tests for boundary conditions and unusual inputs.

use std::time::{Duration, Instant};

use rsnn_eta::RsnnEta;

#[test]
fn test_single_step_total() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .burn_in_ticks(1)
        .seed(42)
        .build();

    let start = Instant::now();
    eta.tick(0, 1, Duration::from_millis(0), start);
    eta.tick(1, 1, Duration::from_millis(100), start + Duration::from_millis(100));

    // Completed — ETA should be ~0
    if let Some(remaining) = eta.last_eta() {
        assert!(remaining.as_secs_f64() < 1.0, "should be near 0 when complete");
    }
}

#[test]
fn test_position_equals_length() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .burn_in_ticks(1)
        .seed(42)
        .build();

    let start = Instant::now();
    for i in 1..=10 {
        eta.tick(i * 10, 100, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    if let Some(remaining) = eta.last_eta() {
        assert!(remaining.as_secs_f64() < 1.0, "should be ~0 at 100%");
    }
}

#[test]
fn test_position_exceeds_length() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .burn_in_ticks(1)
        .seed(42)
        .build();

    let start = Instant::now();
    for i in 1..=5 {
        eta.tick(i * 10, 30, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    // Position 50 > length 30 — should not panic, ETA should be ~0
    if let Some(remaining) = eta.last_eta() {
        assert!(remaining.as_secs_f64() < 1.0, "should be ~0 when past 100%");
    }
}

#[test]
fn test_zero_length_never_panics() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .seed(42)
        .build();

    let start = Instant::now();
    for i in 1..=10 {
        let result = eta.tick(i, 0, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
        assert!(result.is_none(), "should return None with zero length");
    }
}

#[test]
fn test_very_large_position_values() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .burn_in_ticks(1)
        .seed(42)
        .build();

    let start = Instant::now();
    let total = 1_000_000_000u64;
    for i in 1..=10 {
        let pos = i * 100_000_000;
        eta.tick(pos, total, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    assert!(eta.last_eta().is_some());
    let secs = eta.last_eta().unwrap().as_secs_f64();
    assert!(secs.is_finite(), "ETA should be finite with large values");
}

#[test]
fn test_very_small_time_deltas() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .burn_in_ticks(1)
        .seed(42)
        .build();

    let start = Instant::now();
    // Microsecond-level deltas
    for i in 1..=20 {
        let micros = i * 50; // 50μs apart
        eta.tick(i, 1000, Duration::from_micros(micros), start + Duration::from_micros(micros));
    }

    assert!(eta.last_eta().is_some());
    let secs = eta.last_eta().unwrap().as_secs_f64();
    assert!(secs.is_finite() && secs >= 0.0, "ETA should be valid, got {secs}s");
}

#[test]
fn test_minimum_neuron_count() {
    // Even 2 neurons should work (1 reservoir + 1 output)
    let mut eta = RsnnEta::builder()
        .neurons(2)
        .steps_per_tick(5)
        .burn_in_ticks(1)
        .seed(42)
        .build();

    let start = Instant::now();
    for i in 1..=10 {
        eta.tick(i * 10, 1000, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }
    // Should not panic
    assert!(eta.last_eta().is_some());
}

#[test]
fn test_large_neuron_count() {
    let mut eta = RsnnEta::builder()
        .neurons(500)
        .steps_per_tick(5)
        .burn_in_ticks(1)
        .seed(42)
        .build();

    let start = Instant::now();
    for i in 1..=10 {
        eta.tick(i * 10, 1000, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }
    assert!(eta.last_eta().is_some());
}

#[test]
fn test_single_simulation_step() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(1)
        .burn_in_ticks(1)
        .seed(42)
        .build();

    let start = Instant::now();
    for i in 1..=10 {
        eta.tick(i * 10, 1000, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }
    assert!(eta.last_eta().is_some());
}

#[test]
fn test_repeated_reset_cycle() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .burn_in_ticks(1)
        .seed(42)
        .build();

    let start = Instant::now();

    for cycle in 0..3 {
        for i in 1..=10 {
            let t = (cycle * 10 + i) as u64;
            eta.tick(i * 10, 1000, Duration::from_millis(t * 100), start + Duration::from_millis(t * 100));
        }
        assert!(eta.last_eta().is_some(), "cycle {cycle} should produce ETA");
        eta.reset();
        assert!(eta.last_eta().is_none(), "reset should clear ETA");
    }
}
