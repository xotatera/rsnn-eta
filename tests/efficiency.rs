//! Overhead-per-accuracy scoring: measures the tradeoff between tick latency
//! and prediction accuracy across configurations and workload patterns.

use std::time::{Duration, Instant};

use rsnn_eta::{BaseEstimator, EmaEstimator, RsnnEta};

// ── Workload generators ──

fn constant_rate(length: u64, ticks: u64) -> Vec<(u64, u64)> {
    let total_ms = 10_000u64;
    (1..=ticks)
        .map(|i| {
            let pos = (i as f64 / ticks as f64 * length as f64) as u64;
            let ms = (i as f64 / ticks as f64 * total_ms as f64) as u64;
            (pos, ms)
        })
        .collect()
}

fn phase_transition(length: u64) -> Vec<(u64, u64)> {
    let half = length / 2;
    let mut s = Vec::new();
    for i in 1..=50 {
        s.push(((i as f64 / 50.0 * half as f64) as u64, (i as f64 / 50.0 * 5000.0) as u64));
    }
    for i in 1..=50 {
        s.push((half + (i as f64 / 50.0 * half as f64) as u64, 5000 + (i as f64 / 50.0 * 50000.0) as u64));
    }
    s
}

fn bursty(length: u64) -> Vec<(u64, u64)> {
    let step = length / 100;
    let mut s = Vec::new();
    let mut pos = 0u64;
    let mut ms = 0u64;
    for i in 0..100 {
        pos += step;
        ms += if i % 2 == 0 { 10 } else { 500 };
        s.push((pos, ms));
    }
    s
}

fn accelerating(length: u64) -> Vec<(u64, u64)> {
    let step = length / 100;
    let mut s = Vec::new();
    let mut ms = 0u64;
    for i in 1..=100 {
        ms += (500.0 * (1.0 - i as f64 / 100.0)).max(5.0) as u64;
        s.push((i * step, ms));
    }
    s
}

fn decelerating(length: u64) -> Vec<(u64, u64)> {
    let step = length / 100;
    let mut s = Vec::new();
    let mut ms = 0u64;
    for i in 1..=100 {
        ms += (5.0 + 495.0 * (i as f64 / 100.0)) as u64;
        s.push((i * step, ms));
    }
    s
}

fn sawtooth(length: u64) -> Vec<(u64, u64)> {
    let step = length / 100;
    let mut s = Vec::new();
    let mut ms = 0u64;
    for i in 1..=100 {
        // 3 cycles of fast→slow
        let phase = (i % 33) as f64 / 33.0;
        ms += (10.0 + 490.0 * phase) as u64;
        s.push((i * step, ms));
    }
    s
}

// ── Measurement ──

struct RunResult {
    mae: f64,
    median_tick_ns: u64,
    predictions: usize,
}

fn measure_rsnn(
    neurons: usize,
    steps: u32,
    schedule: &[(u64, u64)],
    length: u64,
) -> RunResult {
    let mut eta = RsnnEta::builder()
        .neurons(neurons)
        .steps_per_tick(steps)
        .burn_in_ticks(3)
        .seed(42)
        .build();

    let sim_start = Instant::now();
    let total_elapsed_ms = schedule.last().unwrap().1;
    let total_elapsed = Duration::from_millis(total_elapsed_ms);

    let mut total_error = 0.0;
    let mut count = 0;
    let mut tick_times = Vec::with_capacity(schedule.len());

    for &(pos, elapsed_ms) in schedule {
        let elapsed = Duration::from_millis(elapsed_ms);
        let now = sim_start + elapsed;

        let t0 = Instant::now();
        let result = eta.tick(pos, length, elapsed, now);
        let tick_ns = t0.elapsed().as_nanos() as u64;
        tick_times.push(tick_ns);

        if let Some(predicted) = result {
            let actual_remaining = total_elapsed.saturating_sub(elapsed);
            total_error += (predicted.as_secs_f64() - actual_remaining.as_secs_f64()).abs();
            count += 1;
        }
    }

    tick_times.sort_unstable();
    let median_tick_ns = tick_times[tick_times.len() / 2];
    let mae = if count > 0 { total_error / count as f64 } else { f64::INFINITY };

    RunResult { mae, median_tick_ns, predictions: count }
}

fn measure_ema(schedule: &[(u64, u64)], length: u64) -> RunResult {
    let mut ema = EmaEstimator::new(0.05, 2);
    let total_elapsed_ms = schedule.last().unwrap().1;
    let total_elapsed = Duration::from_millis(total_elapsed_ms);

    let mut total_error = 0.0;
    let mut count = 0;
    let mut tick_times = Vec::with_capacity(schedule.len());

    for &(pos, elapsed_ms) in schedule {
        let elapsed = Duration::from_millis(elapsed_ms);

        let t0 = Instant::now();
        ema.update(pos, length, elapsed);
        let result = ema.estimate();
        let tick_ns = t0.elapsed().as_nanos() as u64;
        tick_times.push(tick_ns);

        if let Some(predicted) = result {
            let actual_remaining = total_elapsed.saturating_sub(elapsed);
            total_error += (predicted.as_secs_f64() - actual_remaining.as_secs_f64()).abs();
            count += 1;
        }
    }

    tick_times.sort_unstable();
    let median_tick_ns = tick_times[tick_times.len() / 2];
    let mae = if count > 0 { total_error / count as f64 } else { f64::INFINITY };

    RunResult { mae, median_tick_ns, predictions: count }
}

// ── Scoring ──

/// Efficiency score: lower is better.
/// Score = MAE * log2(tick_time_ns)
/// This penalizes both inaccuracy and overhead, with diminishing penalty
/// for overhead (going from 1µs to 2µs matters less than 10ns to 20ns).
fn efficiency_score(mae: f64, tick_ns: u64) -> f64 {
    if mae == f64::INFINITY || tick_ns == 0 {
        return f64::INFINITY;
    }
    mae * (tick_ns as f64).log2()
}

#[test]
fn test_efficiency_matrix() {
    let length = 1000u64;

    let workloads: Vec<(&str, Vec<(u64, u64)>)> = vec![
        ("Constant", constant_rate(length, 100)),
        ("Phase shift", phase_transition(length)),
        ("Bursty", bursty(length)),
        ("Accelerating", accelerating(length)),
        ("Decelerating", decelerating(length)),
        ("Sawtooth", sawtooth(length)),
    ];

    let configs: Vec<(&str, usize, u32)> = vec![
        ("EMA only", 0, 0),
        ("10n/5s", 10, 5),
        ("20n/10s", 20, 10),
        ("50n/20s", 50, 20),
        ("100n/20s", 100, 20),
        ("50n/50s", 50, 50),
    ];

    // Header
    println!();
    print!("{:<14}", "Config");
    for (name, _) in &workloads {
        print!(" {:>22}", name);
    }
    println!(" {:>12}", "Avg Score");
    println!("{}", "-".repeat(14 + workloads.len() * 23 + 13));

    for &(config_name, neurons, steps) in &configs {
        print!("{:<14}", config_name);
        let mut scores = Vec::new();

        for (_, schedule) in &workloads {
            let (mae, tick_ns) = if neurons == 0 {
                let r = measure_ema(schedule, length);
                (r.mae, r.median_tick_ns)
            } else {
                let r = measure_rsnn(neurons, steps, schedule, length);
                (r.mae, r.median_tick_ns)
            };

            let score = efficiency_score(mae, tick_ns);
            scores.push(score);

            let tick_str = if tick_ns < 1000 {
                format!("{}ns", tick_ns)
            } else if tick_ns < 1_000_000 {
                format!("{:.1}µs", tick_ns as f64 / 1000.0)
            } else {
                format!("{:.1}ms", tick_ns as f64 / 1_000_000.0)
            };

            print!(" {:>6.2}s {:>6} {:>5.0}", mae, tick_str, score);
        }

        let avg_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        println!(" {:>12.1}", avg_score);
    }

    println!();
    println!("Score = MAE * log2(tick_ns). Lower is better.");
    println!("Columns: MAE(s) | median tick | score");
    println!();
}

/// Focused test: find the pareto-optimal configuration across all workloads.
#[test]
fn test_pareto_optimal() {
    let length = 1000u64;

    let workloads: Vec<Vec<(u64, u64)>> = vec![
        constant_rate(length, 100),
        phase_transition(length),
        bursty(length),
        accelerating(length),
        decelerating(length),
        sawtooth(length),
    ];

    let configs: Vec<(usize, u32)> = vec![
        (10, 5), (10, 10), (10, 20),
        (20, 5), (20, 10), (20, 20),
        (50, 10), (50, 20), (50, 50),
        (100, 10), (100, 20),
    ];

    let mut results: Vec<(usize, u32, f64, f64, f64)> = Vec::new(); // neurons, steps, avg_mae, avg_tick_us, avg_score

    for &(neurons, steps) in &configs {
        let mut total_mae = 0.0;
        let mut total_tick = 0u64;
        let mut total_score = 0.0;

        for schedule in &workloads {
            let r = measure_rsnn(neurons, steps, schedule, length);
            total_mae += r.mae;
            total_tick += r.median_tick_ns;
            total_score += efficiency_score(r.mae, r.median_tick_ns);
        }

        let n = workloads.len() as f64;
        results.push((
            neurons,
            steps,
            total_mae / n,
            total_tick as f64 / n / 1000.0, // µs
            total_score / n,
        ));
    }

    // Also measure EMA baseline
    let mut ema_mae = 0.0;
    let mut ema_tick = 0u64;
    let mut ema_score = 0.0;
    for schedule in &workloads {
        let r = measure_ema(schedule, length);
        ema_mae += r.mae;
        ema_tick += r.median_tick_ns;
        ema_score += efficiency_score(r.mae, r.median_tick_ns);
    }
    let n = workloads.len() as f64;

    // Sort by score
    results.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());

    println!();
    println!("{:<12} {:>10} {:>10} {:>10}", "Config", "Avg MAE(s)", "Avg Tick", "Avg Score");
    println!("{}", "-".repeat(45));
    println!("{:<12} {:>10.2} {:>10} {:>10.1}", "EMA only",
        ema_mae / n,
        format!("{:.0}ns", ema_tick as f64 / n),
        ema_score / n);

    for (neurons, steps, avg_mae, avg_tick_us, avg_score) in &results {
        let tick_str = if *avg_tick_us < 1.0 {
            format!("{:.0}ns", avg_tick_us * 1000.0)
        } else if *avg_tick_us < 1000.0 {
            format!("{:.1}µs", avg_tick_us)
        } else {
            format!("{:.1}ms", avg_tick_us / 1000.0)
        };
        println!("{:<12} {:>10.2} {:>10} {:>10.1}",
            format!("{}n/{}s", neurons, steps), avg_mae, tick_str, avg_score);
    }

    println!();
    println!("Best RSNN config: {}n/{}s (score {:.1})",
        results[0].0, results[0].1, results[0].4);

    let best_rsnn_score = results[0].4;
    let ema_avg_score = ema_score / n;

    // The best RSNN config should not be dramatically worse than EMA
    // on average across all workloads
    assert!(best_rsnn_score < ema_avg_score * 5.0,
        "best RSNN should be within 5x of EMA efficiency: RSNN={best_rsnn_score:.1}, EMA={ema_avg_score:.1}");

    println!("EMA avg score: {:.1}, Best RSNN avg score: {:.1}, ratio: {:.2}x",
        ema_avg_score, best_rsnn_score, best_rsnn_score / ema_avg_score);
    println!();
}
