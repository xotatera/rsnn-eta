use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};
use rsnn_eta::RsnnEta;

fn bench_tick_50_neurons(c: &mut Criterion) {
    let mut eta = RsnnEta::builder()
        .neurons(50)
        .steps_per_tick(20)
        .seed(42)
        .build();

    let start = Instant::now();
    for i in 1..=20 {
        eta.tick(
            i * 10, 10000,
            Duration::from_millis(i * 50),
            start + Duration::from_millis(i * 50),
        );
    }

    let mut tick_num = 21u64;
    c.bench_function("tick_50n_20s", |b| {
        b.iter(|| {
            let elapsed = Duration::from_millis(tick_num * 50);
            let now = start + elapsed;
            eta.tick(tick_num * 10, 100000, elapsed, now);
            tick_num += 1;
        })
    });
}

fn bench_tick_100_neurons(c: &mut Criterion) {
    let mut eta = RsnnEta::builder()
        .neurons(100)
        .steps_per_tick(20)
        .seed(42)
        .build();

    let start = Instant::now();
    for i in 1..=20 {
        eta.tick(
            i * 10, 10000,
            Duration::from_millis(i * 50),
            start + Duration::from_millis(i * 50),
        );
    }

    let mut tick_num = 21u64;
    c.bench_function("tick_100n_20s", |b| {
        b.iter(|| {
            let elapsed = Duration::from_millis(tick_num * 50);
            let now = start + elapsed;
            eta.tick(tick_num * 10, 100000, elapsed, now);
            tick_num += 1;
        })
    });
}

criterion_group!(benches, bench_tick_50_neurons, bench_tick_100_neurons);
criterion_main!(benches);
