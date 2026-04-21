//! Regression tests for bugs found in production.

use std::time::{Duration, Instant};

use rsnn_eta::RsnnEta;

/// Regression: EMA seeding relied on tick_count == 1, so if early calls had
/// dp == 0 (parallel threads reading same position), the EMA was never seeded
/// and stayed at 0.0 forever. Fixed by seeding when ema_step_duration == 0.0.
#[test]
fn test_ema_seeds_despite_zero_delta_calls() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .burn_in_ticks(2)
        .seed(42)
        .build();

    let start = Instant::now();

    // Simulate parallel threads reading the same atomic counter:
    // several calls with the same position before position advances.
    eta.tick(0, 1000, Duration::from_millis(0), start);
    eta.tick(0, 1000, Duration::from_millis(50), start + Duration::from_millis(50));
    eta.tick(0, 1000, Duration::from_millis(100), start + Duration::from_millis(100));
    // Now position advances
    eta.tick(10, 1000, Duration::from_millis(200), start + Duration::from_millis(200));
    eta.tick(20, 1000, Duration::from_millis(300), start + Duration::from_millis(300));
    eta.tick(30, 1000, Duration::from_millis(400), start + Duration::from_millis(400));

    let result = eta.last_eta();
    assert!(result.is_some(), "ETA should be available after position advances");
    let secs = result.unwrap().as_secs_f64();
    assert!(secs > 0.0, "ETA should be positive, got {secs}s");
}

/// Regression: bursty parallel evaluations produce near-zero dt between ticks.
/// The EMA should still converge to a reasonable estimate.
#[test]
fn test_bursty_parallel_evaluations() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .burn_in_ticks(2)
        .seed(42)
        .build();

    let start = Instant::now();
    let total = 1000u64;

    // Phase 1: slow real evaluations (100ms each)
    for i in 1..=10 {
        eta.tick(i, total, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    // Phase 2: burst of cached results (near-instant, 1ms apart)
    let burst_start = 1000u64; // ms
    for i in 11..=50 {
        let ms = burst_start + (i - 10);
        eta.tick(i, total, Duration::from_millis(ms), start + Duration::from_millis(ms));
    }

    // Phase 3: back to slow
    for i in 51..=60 {
        let ms = 1040 + (i - 50) * 100;
        eta.tick(i, total, Duration::from_millis(ms), start + Duration::from_millis(ms));
    }

    let result = eta.last_eta();
    assert!(result.is_some());
    // ETA should be positive and finite
    let secs = result.unwrap().as_secs_f64();
    assert!(secs > 0.0 && secs.is_finite(), "ETA should be positive and finite, got {secs}s");
}

/// The initial message should not stay "--:--" forever even when position
/// doesn't move for a while.
#[test]
fn test_stalled_then_resumed_progress() {
    let mut eta = RsnnEta::builder()
        .neurons(10)
        .steps_per_tick(5)
        .burn_in_ticks(2)
        .seed(42)
        .build();

    let start = Instant::now();
    let total = 100u64;

    // Quick initial progress
    for i in 1..=5 {
        eta.tick(i, total, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    // Stall: position doesn't move for 10 ticks
    for i in 6..=15 {
        eta.tick(5, total, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    // Resume
    for i in 16..=25 {
        let pos = 5 + (i - 15);
        eta.tick(pos, total, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    assert!(eta.last_eta().is_some(), "ETA should recover after stall");
}
