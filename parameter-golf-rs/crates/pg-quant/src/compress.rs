use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use pg_core::error::{PgError, PgResult};
use std::io::{Cursor, Read, Write};

const PERGROUP_MAGIC: &[u8; 4] = b"PGPG";
const RAW_PGRS_MAGIC: &[u8; 4] = b"PGRS";
const PERGROUP_VERSION: u32 = 1;
const DEFAULT_PERGROUP_CHUNK_BYTES: usize = 1 << 20;

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

/// Compress a serialized artifact as independently compressed byte groups.
///
/// This is intentionally an artifact-container transform, not a model-format
/// transform: loaders recover the exact serialized PGRS payload before tensor
/// deserialization. Independent groups let record packaging tune compression
/// locality without changing tensor CRC coverage or quantized tensor semantics.
pub fn compress_pergroup(data: &[u8], level: i32) -> PgResult<Vec<u8>> {
    let chunk_bytes = std::env::var("PG_QUANT_PERGROUP_CHUNK_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_PERGROUP_CHUNK_BYTES);
    let group_count = data.len().div_ceil(chunk_bytes);
    let mut out = Vec::with_capacity(data.len());
    out.write_all(PERGROUP_MAGIC)?;
    out.write_u32::<LittleEndian>(PERGROUP_VERSION)?;
    out.write_u64::<LittleEndian>(data.len() as u64)?;
    out.write_u32::<LittleEndian>(chunk_bytes as u32)?;
    out.write_u32::<LittleEndian>(group_count as u32)?;
    for chunk in data.chunks(chunk_bytes) {
        let compressed = zstd::encode_all(chunk, level)?;
        out.write_u32::<LittleEndian>(chunk.len() as u32)?;
        out.write_u32::<LittleEndian>(compressed.len() as u32)?;
        out.write_all(&compressed)?;
    }
    Ok(out)
}

pub fn is_pergroup_artifact(data: &[u8]) -> bool {
    data.starts_with(PERGROUP_MAGIC)
}

pub fn decompress_pergroup(data: &[u8]) -> PgResult<Vec<u8>> {
    let mut cursor = Cursor::new(data);
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != PERGROUP_MAGIC {
        return Err(PgError::DataFormat(
            "wrong pergroup artifact magic bytes".into(),
        ));
    }
    let version = cursor.read_u32::<LittleEndian>()?;
    if version != PERGROUP_VERSION {
        return Err(PgError::DataFormat(format!(
            "unsupported pergroup artifact version: {version}"
        )));
    }
    let raw_len = cursor.read_u64::<LittleEndian>()? as usize;
    let _chunk_bytes = cursor.read_u32::<LittleEndian>()? as usize;
    let group_count = cursor.read_u32::<LittleEndian>()? as usize;
    let mut out = Vec::with_capacity(raw_len);
    for _ in 0..group_count {
        let expected_raw = cursor.read_u32::<LittleEndian>()? as usize;
        let compressed_len = cursor.read_u32::<LittleEndian>()? as usize;
        let mut compressed = vec![0u8; compressed_len];
        cursor.read_exact(&mut compressed)?;
        let chunk = zstd::decode_all(compressed.as_slice())?;
        if chunk.len() != expected_raw {
            return Err(PgError::DataFormat(format!(
                "pergroup chunk length mismatch: expected {expected_raw}, got {}",
                chunk.len()
            )));
        }
        out.extend_from_slice(&chunk);
    }
    if out.len() != raw_len {
        return Err(PgError::DataFormat(format!(
            "pergroup artifact length mismatch: expected {raw_len}, got {}",
            out.len()
        )));
    }
    Ok(out)
}

pub fn decompress_artifact_payload(data: &[u8]) -> PgResult<Vec<u8>> {
    if is_pergroup_artifact(data) {
        decompress_pergroup(data)
    } else if data.starts_with(RAW_PGRS_MAGIC) {
        Ok(data.to_vec())
    } else {
        decompress_zstd(data)
    }
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

    #[test]
    fn test_pergroup_roundtrip() {
        let original: Vec<u8> = (0..50_000).map(|i| (i % 251) as u8).collect();
        let compressed = compress_pergroup(&original, 22).unwrap();
        assert!(is_pergroup_artifact(&compressed));
        let decompressed = decompress_artifact_payload(&compressed).unwrap();
        assert_eq!(original, decompressed);
    }
}
