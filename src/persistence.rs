use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::decoder::Decoder;
use crate::network::SnnNetwork;
use crate::tracker::RsnnEtaCore;

/// Serializable snapshot of learnable state.
#[derive(Serialize, Deserialize)]
struct Snapshot {
    network: SnnNetwork,
    decoder: Decoder,
}

/// Save learnable state to disk.
pub fn save(core: &RsnnEtaCore, path: &Path) -> io::Result<()> {
    let snapshot = Snapshot {
        network: core.network.clone(),
        decoder: core.decoder.clone(),
    };
    let bytes = bincode::serialize(&snapshot)
        .map_err(io::Error::other)?;
    std::fs::write(path, bytes)
}

/// Load learnable state from disk into an existing core.
pub fn load(core: &mut RsnnEtaCore, path: &Path) -> io::Result<()> {
    let bytes = std::fs::read(path)?;
    let snapshot: Snapshot = bincode::deserialize(&bytes)
        .map_err(io::Error::other)?;
    core.network = snapshot.network;
    core.decoder = snapshot.decoder;
    core.stdp = crate::stdp::StdpState::new(&core.network, core.stdp.config.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DecoderConfig, NetworkConfig, StdpConfig};
    use crate::estimator::EmaEstimator;
    use std::time::{Duration, Instant};

    fn make_core() -> RsnnEtaCore {
        RsnnEtaCore::new(
            NetworkConfig::default(),
            StdpConfig::default(),
            DecoderConfig::default(),
            Box::new(EmaEstimator::default()),
            10,
            42,
        )
    }

    #[test]
    fn test_round_trip() {
        let mut core = make_core();
        let start = Instant::now();
        for i in 1..=20 {
            core.tick(i * 5, 1000, Duration::from_millis(i * 50), start + Duration::from_millis(i * 50));
        }

        let path = std::env::temp_dir().join("rsnn_eta_test_roundtrip.bin");
        save(&core, &path).unwrap();

        let mut core2 = make_core();
        load(&mut core2, &path).unwrap();

        assert_eq!(core.network.neurons.len(), core2.network.neurons.len());
        for (a, b) in core.network.recurrent_synapses.iter()
            .zip(core2.network.recurrent_synapses.iter())
        {
            for (sa, sb) in a.iter().zip(b.iter()) {
                assert!((sa.weight - sb.weight).abs() < 1e-15,
                    "weight mismatch: {} vs {}", sa.weight, sb.weight);
            }
        }

        assert!((core.decoder.scale - core2.decoder.scale).abs() < 1e-15);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_fails() {
        let mut core = make_core();
        let result = load(&mut core, Path::new("/tmp/nonexistent_rsnn_eta.bin"));
        assert!(result.is_err());
    }
}
