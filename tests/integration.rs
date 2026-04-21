use std::time::{Duration, Instant};

use rsnn_eta::RsnnEta;

#[test]
fn test_constant_rate_produces_reasonable_eta() {
    let mut eta = RsnnEta::builder()
        .neurons(30)
        .steps_per_tick(10)
        .burn_in_ticks(5)
        .seed(123)
        .build();

    let start = Instant::now();
    let total_steps = 1000u64;
    let steps_per_tick = 10u64;

    for tick in 1..=50 {
        let pos = tick * steps_per_tick;
        let elapsed = Duration::from_millis(tick * 100);
        let now = start + elapsed;
        eta.tick(pos, total_steps, elapsed, now);
    }

    let last_eta = eta.last_eta().unwrap();
    let secs = last_eta.as_secs_f64();
    assert!(secs > 1.0 && secs < 20.0,
        "expected ~5s ETA for constant rate, got {secs}s");
}

#[test]
fn test_phase_transition_adapts() {
    let mut eta = RsnnEta::builder()
        .neurons(30)
        .steps_per_tick(10)
        .burn_in_ticks(3)
        .seed(456)
        .build();

    let start = Instant::now();
    let total = 2000u64;

    // Phase 1: fast (100 steps/sec)
    for tick in 1..=30 {
        let pos = tick * 10;
        let elapsed = Duration::from_millis(tick * 100);
        eta.tick(pos, total, elapsed, start + elapsed);
    }
    let eta_before = eta.last_eta();

    // Phase 2: slow (1 step/sec)
    for tick in 31..=60 {
        let offset = tick - 30;
        let pos = 300 + offset;
        let elapsed = Duration::from_millis(3000 + offset * 1000);
        eta.tick(pos, total, elapsed, start + elapsed);
    }

    assert!(eta.last_eta().is_some());
    if let (Some(before), Some(after)) = (eta_before, eta.last_eta()) {
        assert!(after > before,
            "ETA should increase after slowdown: before={:?}, after={:?}", before, after);
    }
}

#[test]
fn test_cold_start_returns_none_then_some() {
    let mut eta = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(5)
        .burn_in_ticks(5)
        .seed(789)
        .build();

    let start = Instant::now();
    let result = eta.tick(1, 100, Duration::from_millis(100), start + Duration::from_millis(100));
    assert!(result.is_none());

    for i in 2..=20 {
        eta.tick(i * 5, 1000, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }
    assert!(eta.last_eta().is_some());
}

#[test]
fn test_side_channel_signals() {
    let (mut eta, tx) = RsnnEta::builder()
        .neurons(30)
        .steps_per_tick(10)
        .seed(111)
        .build_with_signals();

    let start = Instant::now();
    tx.send(vec![0.5]).unwrap();

    for i in 1..=20 {
        eta.tick(i * 10, 1000, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }

    assert_eq!(eta.core.encoder.input_dim, 5); // 4 core + 1 side channel
}

#[test]
fn test_persistence_round_trip() {
    let path = std::env::temp_dir().join("rsnn_eta_integration_test.bin");

    let mut eta1 = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(5)
        .persistence(path.clone())
        .seed(42)
        .build();

    let start = Instant::now();
    for i in 1..=30 {
        eta1.tick(i * 10, 1000, Duration::from_millis(i * 100), start + Duration::from_millis(i * 100));
    }
    eta1.save().unwrap();

    let eta2 = RsnnEta::builder()
        .neurons(20)
        .steps_per_tick(5)
        .persistence(path.clone())
        .seed(99)
        .build();

    assert!((eta1.core.decoder.scale - eta2.core.decoder.scale).abs() < 1e-15);

    let _ = std::fs::remove_file(&path);
}
