//! Tests verifying thread-safe usage patterns (Arc<Mutex<RsnnEta>>).

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rsnn_eta::RsnnEta;

/// Simulate the parallel evolution pattern from insights: multiple threads
/// increment a shared counter and call tick() behind a Mutex.
#[test]
fn test_arc_mutex_parallel_usage() {
    let eta = Arc::new(Mutex::new(
        RsnnEta::builder()
            .neurons(20)
            .steps_per_tick(5)
            .burn_in_ticks(2)
            .seed(42)
            .build()
    ));
    let counter = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let total = 200u64;

    let handles: Vec<_> = (0..4).map(|_| {
        let eta = eta.clone();
        let counter = counter.clone();
        std::thread::spawn(move || {
            for _ in 0..50 {
                let pos = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if let Ok(mut eta) = eta.lock() {
                    let elapsed = start.elapsed();
                    eta.tick(pos, total, elapsed, Instant::now());
                }
                // Simulate some work
                std::thread::sleep(Duration::from_micros(10));
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    let eta = eta.lock().unwrap();
    assert!(eta.last_eta().is_some(), "should have ETA after parallel ticks");
    let secs = eta.last_eta().unwrap().as_secs_f64();
    assert!(secs.is_finite() && secs >= 0.0, "ETA should be valid, got {secs}s");
}

/// Concurrent ticks should not produce NaN or infinite ETAs.
#[test]
fn test_no_nan_under_contention() {
    let eta = Arc::new(Mutex::new(
        RsnnEta::builder()
            .neurons(10)
            .steps_per_tick(3)
            .burn_in_ticks(1)
            .seed(99)
            .build()
    ));
    let counter = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..8).map(|_| {
        let eta = eta.clone();
        let counter = counter.clone();
        std::thread::spawn(move || {
            for _ in 0..25 {
                let pos = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if let Ok(mut eta) = eta.lock() {
                    let elapsed = start.elapsed();
                    if let Some(remaining) = eta.tick(pos, 500, elapsed, Instant::now()) {
                        let secs = remaining.as_secs_f64();
                        assert!(!secs.is_nan(), "ETA should never be NaN");
                        assert!(secs.is_finite(), "ETA should be finite");
                    }
                }
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }
}

/// Simulate the actual evolution pattern: some threads finish instantly
/// (cached genomes), others take real time (fresh evaluations).
#[test]
fn test_mixed_fast_slow_parallel() {
    let eta = Arc::new(Mutex::new(
        RsnnEta::builder()
            .neurons(15)
            .steps_per_tick(5)
            .burn_in_ticks(2)
            .seed(42)
            .build()
    ));
    let counter = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let total = 100u64;

    let handles: Vec<_> = (0..4).map(|thread_id| {
        let eta = eta.clone();
        let counter = counter.clone();
        std::thread::spawn(move || {
            for i in 0..25 {
                let pos = counter.fetch_add(1, Ordering::Relaxed) + 1;

                // Odd threads are "fast" (cached), even are "slow" (real eval)
                if thread_id % 2 == 0 {
                    std::thread::sleep(Duration::from_millis(5));
                }

                if let Ok(mut eta) = eta.lock() {
                    let elapsed = start.elapsed();
                    eta.tick(pos, total, elapsed, Instant::now());
                }
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    let eta = eta.lock().unwrap();
    assert!(eta.last_eta().is_some(), "should produce ETA with mixed fast/slow threads");
}
