use pg_core::error::PgResult;

/// Compress data with zstd at level 22.
///
/// zstd-22 compression ratios on quantized weights:
/// - Int5 MLP:       ~1.88× (4.26 effective bits/param)
/// - Int6 attention:  ~1.51× (5.30 effective bits/param)
/// - Int8 embeddings: ~1.25× (6.40 effective bits/param)
pub fn compress_zstd(data: &[u8], level: i32) -> PgResult<Vec<u8>> {
    let compressed = zstd::encode_all(data, level)?;
    Ok(compressed)
}

/// Decompress zstd data.
pub fn decompress_zstd(data: &[u8]) -> PgResult<Vec<u8>> {
    let decompressed = zstd::decode_all(data)?;
    Ok(decompressed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let original = b"hello world, this is a test of zstd compression!";
        let compressed = compress_zstd(original, 22).unwrap();
        let decompressed = decompress_zstd(&compressed).unwrap();
        assert_eq!(original.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_quantized_data_compression() {
        // Simulate int6 quantized weights: values in [-32, 31]
        let n = 512 * 512;
        let data: Vec<u8> = (0..n)
            .map(|i| ((i as f32 * 0.01).sin() * 31.0) as i8 as u8)
            .collect();

        let compressed = compress_zstd(&data, 22).unwrap();
        let ratio = data.len() as f64 / compressed.len() as f64;
        eprintln!(
            "zstd-22: {} bytes → {} bytes ({:.2}× ratio)",
            data.len(),
            compressed.len(),
            ratio
        );

        let decompressed = decompress_zstd(&compressed).unwrap();
        assert_eq!(data, decompressed);
    }
}
