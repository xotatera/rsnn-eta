use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rsnn_eta::{BaseEstimator, EmaEstimator, RsnnEta};

// ── Helpers ──

fn warmup_eta(eta: &mut RsnnEta, ticks: u64, start: Instant) {
    for i in 1..=ticks {
        eta.tick(
            i * 10,
            100_000,
            Duration::from_millis(i * 50),
            start + Duration::from_millis(i * 50),
        );
    }
}

fn warmup_ema(ema: &mut EmaEstimator, ticks: u64) {
    for i in 1..=ticks {
        ema.update(i * 10, 100_000, Duration::from_millis(i * 50));
    }
}

// ── Neuron scaling ──

fn bench_neuron_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_scaling");

    for neurons in [10, 20, 50, 100, 200, 500] {
        group.bench_with_input(
            BenchmarkId::new("tick", neurons),
            &neurons,
            |b, &neurons| {
                let mut eta = RsnnEta::builder()
                    .neurons(neurons)
                    .steps_per_tick(20)
                    .seed(42)
                    .build();
                let start = Instant::now();
                warmup_eta(&mut eta, 20, start);

                let mut tick_num = 21u64;
                b.iter(|| {
                    let elapsed = Duration::from_millis(tick_num * 50);
                    black_box(eta.tick(
                        black_box(tick_num * 10),
                        100_000,
                        elapsed,
                        start + elapsed,
                    ));
                    tick_num = tick_num.wrapping_add(1);
                });
            },
        );
    }

    group.finish();
}

// ── Steps-per-tick scaling ──

fn bench_steps_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("steps_scaling");

    for steps in [1, 5, 10, 20, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("tick", steps),
            &steps,
            |b, &steps| {
                let mut eta = RsnnEta::builder()
                    .neurons(50)
                    .steps_per_tick(steps)
                    .seed(42)
                    .build();
                let start = Instant::now();
                warmup_eta(&mut eta, 20, start);

                let mut tick_num = 21u64;
                b.iter(|| {
                    let elapsed = Duration::from_millis(tick_num * 50);
                    black_box(eta.tick(
                        black_box(tick_num * 10),
                        100_000,
                        elapsed,
                        start + elapsed,
                    ));
                    tick_num = tick_num.wrapping_add(1);
                });
            },
        );
    }

    group.finish();
}

// ── RSNN tick vs plain EMA update ──

fn bench_rsnn_vs_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group("rsnn_vs_ema");

    // RSNN tick (50 neurons, 20 steps — default config)
    group.bench_function("rsnn_50n_20s", |b| {
        let mut eta = RsnnEta::builder()
            .neurons(50)
            .steps_per_tick(20)
            .seed(42)
            .build();
        let start = Instant::now();
        warmup_eta(&mut eta, 20, start);

        let mut tick_num = 21u64;
        b.iter(|| {
            let elapsed = Duration::from_millis(tick_num * 50);
            black_box(eta.tick(
                black_box(tick_num * 10),
                100_000,
                elapsed,
                start + elapsed,
            ));
            tick_num = tick_num.wrapping_add(1);
        });
    });

    // Plain EMA (the base estimator alone, no RSNN)
    group.bench_function("ema_only", |b| {
        let mut ema = EmaEstimator::new(0.05, 2);
        warmup_ema(&mut ema, 20);

        let mut tick_num = 21u64;
        b.iter(|| {
            ema.update(
                black_box(tick_num * 10),
                100_000,
                Duration::from_millis(tick_num * 50),
            );
            black_box(ema.estimate());
            tick_num = tick_num.wrapping_add(1);
        });
    });

    group.finish();
}

// ── Construction cost ──

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    for neurons in [20, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("build", neurons),
            &neurons,
            |b, &neurons| {
                b.iter(|| {
                    black_box(RsnnEta::builder()
                        .neurons(neurons)
                        .steps_per_tick(20)
                        .seed(42)
                        .build());
                });
            },
        );
    }

    group.finish();
}

// ── Side-channel overhead ──

fn bench_side_channel(c: &mut Criterion) {
    let mut group = c.benchmark_group("side_channel");

    // Without side channel
    group.bench_function("no_side_channel", |b| {
        let mut eta = RsnnEta::builder()
            .neurons(50)
            .steps_per_tick(20)
            .seed(42)
            .build();
        let start = Instant::now();
        warmup_eta(&mut eta, 20, start);

        let mut tick_num = 21u64;
        b.iter(|| {
            let elapsed = Duration::from_millis(tick_num * 50);
            black_box(eta.tick(
                black_box(tick_num * 10),
                100_000,
                elapsed,
                start + elapsed,
            ));
            tick_num = tick_num.wrapping_add(1);
        });
    });

    // With 3 side-channel signals
    group.bench_function("3_signals", |b| {
        let (mut eta, tx) = RsnnEta::builder()
            .neurons(50)
            .steps_per_tick(20)
            .seed(42)
            .build_with_signals();
        let start = Instant::now();
        // First tick with side channel to expand input dim
        tx.send(vec![0.5, 0.3, 0.8]).unwrap();
        warmup_eta(&mut eta, 20, start);

        let mut tick_num = 21u64;
        b.iter(|| {
            tx.send(vec![0.5, 0.3, 0.8]).unwrap();
            let elapsed = Duration::from_millis(tick_num * 50);
            black_box(eta.tick(
                black_box(tick_num * 10),
                100_000,
                elapsed,
                start + elapsed,
            ));
            tick_num = tick_num.wrapping_add(1);
        });
    });

    group.finish();
}

// ── Mutex contention simulation ──

fn bench_mutex_contention(c: &mut Criterion) {
    use std::sync::{Arc, Mutex};

    let mut group = c.benchmark_group("mutex_contention");

    // Uncontended lock + tick
    group.bench_function("uncontended", |b| {
        let eta = Arc::new(Mutex::new(
            RsnnEta::builder()
                .neurons(50)
                .steps_per_tick(20)
                .seed(42)
                .build(),
        ));
        let start = Instant::now();
        {
            let mut e = eta.lock().unwrap();
            warmup_eta(&mut e, 20, start);
        }

        let mut tick_num = 21u64;
        b.iter(|| {
            let mut e = eta.lock().unwrap();
            let elapsed = Duration::from_millis(tick_num * 50);
            black_box(e.tick(
                black_box(tick_num * 10),
                100_000,
                elapsed,
                start + elapsed,
            ));
            tick_num = tick_num.wrapping_add(1);
        });
    });

    group.finish();
}

// ── Persistence ──

fn bench_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence");
    let path = std::env::temp_dir().join("rsnn_eta_bench_persist.bin");

    // Save
    group.bench_function("save_50n", |b| {
        let mut eta = RsnnEta::builder()
            .neurons(50)
            .steps_per_tick(20)
            .persistence(path.clone())
            .seed(42)
            .build();
        let start = Instant::now();
        warmup_eta(&mut eta, 20, start);

        b.iter(|| {
            black_box(eta.save().unwrap());
        });
    });

    // Load
    {
        let mut eta = RsnnEta::builder()
            .neurons(50)
            .steps_per_tick(20)
            .persistence(path.clone())
            .seed(42)
            .build();
        let start = Instant::now();
        warmup_eta(&mut eta, 20, start);
        eta.save().unwrap();
    }

    group.bench_function("load_50n", |b| {
        let mut eta = RsnnEta::builder()
            .neurons(50)
            .steps_per_tick(20)
            .persistence(path.clone())
            .seed(42)
            .build();

        b.iter(|| {
            black_box(eta.load().unwrap());
        });
    });

    let _ = std::fs::remove_file(&path);
    group.finish();
}

criterion_group!(
    benches,
    bench_neuron_scaling,
    bench_steps_scaling,
    bench_rsnn_vs_ema,
    bench_construction,
    bench_side_channel,
    bench_mutex_contention,
    bench_persistence,
);
criterion_main!(benches);
