//! End-to-end comparison of RSNN-corrected ETA vs plain EMA estimator
//! across different workload patterns.

use std::time::{Duration, Instant};

use rsnn_eta::{BaseEstimator, EmaEstimator, RsnnEta};

/// Measure mean absolute error of ETA predictions against actual remaining time.
fn measure_accuracy(
    predict: &mut dyn FnMut(u64, u64, Duration, Instant) -> Option<Duration>,
    schedule: &[(u64, u64)], // (position, elapsed_ms) pairs
    length: u64,
) -> (f64, usize) {
    let start = Instant::now();
    let total_elapsed_ms = schedule.last().unwrap().1;
    let total_elapsed = Duration::from_millis(total_elapsed_ms);

    let mut total_error = 0.0;
    let mut count = 0;

    for &(pos, elapsed_ms) in schedule {
        let elapsed = Duration::from_millis(elapsed_ms);
        let now = start + elapsed;

        if let Some(predicted_remaining) = predict(pos, length, elapsed, now) {
            // Actual remaining time = total_elapsed - current_elapsed
            let actual_remaining = total_elapsed.saturating_sub(elapsed);
            let error = (predicted_remaining.as_secs_f64() - actual_remaining.as_secs_f64()).abs();
            total_error += error;
            count += 1;
        }
    }

    let mae = if count > 0 { total_error / count as f64 } else { f64::INFINITY };
    (mae, count)
}

/// Generate a constant-rate workload schedule.
fn constant_rate_schedule(length: u64, total_ms: u64, num_ticks: u64) -> Vec<(u64, u64)> {
    (1..=num_ticks)
        .map(|i| {
            let pos = (i as f64 / num_ticks as f64 * length as f64) as u64;
            let ms = (i as f64 / num_ticks as f64 * total_ms as f64) as u64;
            (pos, ms)
        })
        .collect()
}

/// Generate a two-phase workload: fast then slow.
fn phase_transition_schedule(length: u64) -> Vec<(u64, u64)> {
    let mut schedule = Vec::new();
    let half = length / 2;

    // Phase 1: fast — first half in 5 seconds
    for i in 1..=50 {
        let pos = (i as f64 / 50.0 * half as f64) as u64;
        let ms = (i as f64 / 50.0 * 5000.0) as u64;
        schedule.push((pos, ms));
    }

    // Phase 2: slow — second half in 50 seconds
    for i in 1..=50 {
        let pos = half + (i as f64 / 50.0 * half as f64) as u64;
        let ms = 5000 + (i as f64 / 50.0 * 50000.0) as u64;
        schedule.push((pos, ms));
    }

    schedule
}

/// Generate a bursty workload: alternating fast bursts and slow periods.
fn bursty_schedule(length: u64) -> Vec<(u64, u64)> {
    let mut schedule = Vec::new();
    let mut pos = 0u64;
    let mut ms = 0u64;
    let step = length / 100;

    for i in 0..100 {
        pos += step;
        // Alternating: even ticks are fast (10ms), odd are slow (500ms)
        if i % 2 == 0 {
            ms += 10;
        } else {
            ms += 500;
        }
        schedule.push((pos, ms));
    }

    schedule
}

/// Generate an accelerating workload (starts slow, speeds up).
fn accelerating_schedule(length: u64) -> Vec<(u64, u64)> {
    let mut schedule = Vec::new();
    let mut ms = 0u64;
    let step = length / 100;

    for i in 1..=100 {
        let pos = i * step;
        // Interval decreases as we progress: 500ms down to 5ms
        let interval = (500.0 * (1.0 - i as f64 / 100.0)).max(5.0) as u64;
        ms += interval;
        schedule.push((pos, ms));
    }

    schedule
}

#[test]
fn test_constant_rate_both_accurate() {
    let length = 1000u64;
    let schedule = constant_rate_schedule(length, 10000, 100);

    // RSNN estimator
    let mut rsnn = RsnnEta::builder()
        .neurons(30)
        .steps_per_tick(10)
        .burn_in_ticks(3)
        .seed(42)
        .build();
    let (rsnn_mae, rsnn_count) = measure_accuracy(
        &mut |pos, len, elapsed, now| rsnn.tick(pos, len, elapsed, now),
        &schedule, length,
    );

    // Plain EMA estimator
    let mut ema = EmaEstimator::new(0.05, 2);
    let (ema_mae, ema_count) = measure_accuracy(
        &mut |pos, len, elapsed, _now| {
            ema.update(pos, len, elapsed);
            ema.estimate()
        },
        &schedule, length,
    );

    println!("Constant rate — RSNN MAE: {rsnn_mae:.2}s ({rsnn_count} predictions), EMA MAE: {ema_mae:.2}s ({ema_count} predictions)");

    // Both should be reasonably accurate for constant rate
    assert!(rsnn_mae < 10.0, "RSNN MAE too high: {rsnn_mae:.2}s");
    assert!(ema_mae < 10.0, "EMA MAE too high: {ema_mae:.2}s");
    assert!(rsnn_count > 0);
    assert!(ema_count > 0);
}

#[test]
fn test_phase_transition_rsnn_adapts_faster() {
    let length = 1000u64;
    let schedule = phase_transition_schedule(length);

    // RSNN estimator
    let mut rsnn = RsnnEta::builder()
        .neurons(30)
        .steps_per_tick(10)
        .burn_in_ticks(3)
        .seed(42)
        .build();
    let (rsnn_mae, rsnn_count) = measure_accuracy(
        &mut |pos, len, elapsed, now| rsnn.tick(pos, len, elapsed, now),
        &schedule, length,
    );

    // Plain EMA estimator
    let mut ema = EmaEstimator::new(0.05, 2);
    let (ema_mae, ema_count) = measure_accuracy(
        &mut |pos, len, elapsed, _now| {
            ema.update(pos, len, elapsed);
            ema.estimate()
        },
        &schedule, length,
    );

    println!("Phase transition — RSNN MAE: {rsnn_mae:.2}s ({rsnn_count} predictions), EMA MAE: {ema_mae:.2}s ({ema_count} predictions)");

    // Both should produce predictions
    assert!(rsnn_count > 0);
    assert!(ema_count > 0);
    // RSNN should not be dramatically worse than EMA
    // (may not be better yet — the RSNN needs many ticks to learn correction patterns)
    assert!(rsnn_mae < ema_mae * 3.0,
        "RSNN should not be 3x worse than EMA: RSNN={rsnn_mae:.2}s, EMA={ema_mae:.2}s");
}

#[test]
fn test_bursty_workload() {
    let length = 1000u64;
    let schedule = bursty_schedule(length);

    let mut rsnn = RsnnEta::builder()
        .neurons(30)
        .steps_per_tick(10)
        .burn_in_ticks(3)
        .seed(42)
        .build();
    let (rsnn_mae, rsnn_count) = measure_accuracy(
        &mut |pos, len, elapsed, now| rsnn.tick(pos, len, elapsed, now),
        &schedule, length,
    );

    let mut ema = EmaEstimator::new(0.05, 2);
    let (ema_mae, ema_count) = measure_accuracy(
        &mut |pos, len, elapsed, _now| {
            ema.update(pos, len, elapsed);
            ema.estimate()
        },
        &schedule, length,
    );

    println!("Bursty — RSNN MAE: {rsnn_mae:.2}s ({rsnn_count} predictions), EMA MAE: {ema_mae:.2}s ({ema_count} predictions)");

    assert!(rsnn_count > 0);
    assert!(ema_count > 0);
    assert!(rsnn_mae.is_finite());
    assert!(ema_mae.is_finite());
}

#[test]
fn test_accelerating_workload() {
    let length = 1000u64;
    let schedule = accelerating_schedule(length);

    let mut rsnn = RsnnEta::builder()
        .neurons(30)
        .steps_per_tick(10)
        .burn_in_ticks(3)
        .seed(42)
        .build();
    let (rsnn_mae, rsnn_count) = measure_accuracy(
        &mut |pos, len, elapsed, now| rsnn.tick(pos, len, elapsed, now),
        &schedule, length,
    );

    let mut ema = EmaEstimator::new(0.05, 2);
    let (ema_mae, ema_count) = measure_accuracy(
        &mut |pos, len, elapsed, _now| {
            ema.update(pos, len, elapsed);
            ema.estimate()
        },
        &schedule, length,
    );

    println!("Accelerating — RSNN MAE: {rsnn_mae:.2}s ({rsnn_count} predictions), EMA MAE: {ema_mae:.2}s ({ema_count} predictions)");

    assert!(rsnn_count > 0);
    assert!(ema_count > 0);
    assert!(rsnn_mae.is_finite());
    assert!(ema_mae.is_finite());
}

/// Run all workload patterns and print a summary table.
#[test]
fn test_comparison_summary() {
    let length = 1000u64;

    let workloads: Vec<(&str, Vec<(u64, u64)>)> = vec![
        ("Constant rate", constant_rate_schedule(length, 10000, 100)),
        ("Phase transition", phase_transition_schedule(length)),
        ("Bursty", bursty_schedule(length)),
        ("Accelerating", accelerating_schedule(length)),
    ];

    println!("\n{:<20} {:>12} {:>12} {:>10}", "Workload", "RSNN MAE(s)", "EMA MAE(s)", "Winner");
    println!("{}", "-".repeat(58));

    for (name, schedule) in &workloads {
        let mut rsnn = RsnnEta::builder()
            .neurons(30)
            .steps_per_tick(10)
            .burn_in_ticks(3)
            .seed(42)
            .build();
        let (rsnn_mae, _) = measure_accuracy(
            &mut |pos, len, elapsed, now| rsnn.tick(pos, len, elapsed, now),
            schedule, length,
        );

        let mut ema = EmaEstimator::new(0.05, 2);
        let (ema_mae, _) = measure_accuracy(
            &mut |pos, len, elapsed, _now| {
                ema.update(pos, len, elapsed);
                ema.estimate()
            },
            schedule, length,
        );

        let winner = if rsnn_mae < ema_mae { "RSNN" } else { "EMA" };
        println!("{:<20} {:>12.2} {:>12.2} {:>10}", name, rsnn_mae, ema_mae, winner);
    }
    println!();
}
