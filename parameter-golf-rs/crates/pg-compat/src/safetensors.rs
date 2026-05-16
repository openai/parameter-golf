/// Safetensors file reader.
///
/// Safetensors format:
///   [8 bytes: header_size as u64 LE]
///   [header_size bytes: JSON header]
///   [tensor data]
///
/// JSON header maps tensor names to {dtype, shape, data_offsets: [start, end]}.
/// Data offsets are relative to the end of the header.
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use half::{bf16, f16};

/// Tensor metadata from safetensors header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

/// Loaded safetensors file.
pub struct SafeTensorsFile {
    pub tensors: HashMap<String, TensorInfo>,
    pub data: Vec<u8>,  // raw tensor data (after header)
    data_offset: usize, // byte offset where tensor data starts in the file
}

impl SafeTensorsFile {
    /// Load a safetensors file from disk.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes =
            fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
        Self::from_bytes(&bytes)
    }

    /// Parse from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("file too small for safetensors header".into());
        }

        let header_size = (&bytes[..8])
            .read_u64::<LittleEndian>()
            .map_err(|e| format!("failed to read header size: {}", e))?
            as usize;

        if bytes.len() < 8 + header_size {
            return Err(format!(
                "file too small: need {} bytes for header, have {}",
                8 + header_size,
                bytes.len()
            ));
        }

        let header_json = std::str::from_utf8(&bytes[8..8 + header_size])
            .map_err(|e| format!("invalid UTF-8 in header: {}", e))?;

        let tensors = parse_header(header_json)?;
        let data_offset = 8 + header_size;
        let data = bytes[data_offset..].to_vec();

        Ok(Self {
            tensors,
            data,
            data_offset,
        })
    }

    /// Get tensor data as f32 (converting from bf16/f16/f32 as needed).
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor '{}' not found", name))?;

        let raw = &self.data[info.data_start..info.data_end];
        let numel: usize = info.shape.iter().product();

        match info.dtype.as_str() {
            "F32" => {
                if raw.len() != numel * 4 {
                    return Err(format!("F32 size mismatch: {} vs {}", raw.len(), numel * 4));
                }
                let floats: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(floats)
            }
            "BF16" => {
                if raw.len() != numel * 2 {
                    return Err(format!(
                        "BF16 size mismatch: {} vs {}",
                        raw.len(),
                        numel * 2
                    ));
                }
                let floats: Vec<f32> = raw
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        bf16::from_bits(bits).to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            "F16" => {
                if raw.len() != numel * 2 {
                    return Err(format!("F16 size mismatch: {} vs {}", raw.len(), numel * 2));
                }
                let floats: Vec<f32> = raw
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16::from_bits(bits).to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            other => Err(format!("unsupported dtype: {}", other)),
        }
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

/// Parse the JSON header (minimal JSON parser for safetensors format).
/// We avoid pulling in serde_json by parsing the simple structure ourselves.
fn parse_header(json: &str) -> Result<HashMap<String, TensorInfo>, String> {
    // Safetensors header is a JSON object:
    // {"tensor_name": {"dtype": "F32", "shape": [dim1, dim2], "data_offsets": [start, end]}, ...}
    // May also have "__metadata__" key which we skip.

    let mut tensors = HashMap::new();
    let trimmed = json.trim();

    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Err("header is not a JSON object".into());
    }

    // Simple state machine parser
    let chars: Vec<char> = trimmed.chars().collect();
    let mut pos = 1; // skip opening '{'

    loop {
        skip_whitespace(&chars, &mut pos);
        if pos >= chars.len() || chars[pos] == '}' {
            break;
        }

        // Parse key
        let key = parse_json_string(&chars, &mut pos)?;
        skip_whitespace(&chars, &mut pos);

        if pos >= chars.len() || chars[pos] != ':' {
            return Err(format!("expected ':' after key '{}' at pos {}", key, pos));
        }
        pos += 1; // skip ':'
        skip_whitespace(&chars, &mut pos);

        if key == "__metadata__" {
            // Skip metadata value
            skip_json_value(&chars, &mut pos)?;
        } else {
            // Parse tensor info object
            let info = parse_tensor_info(&chars, &mut pos, &key)?;
            tensors.insert(key, info);
        }

        skip_whitespace(&chars, &mut pos);
        if pos < chars.len() && chars[pos] == ',' {
            pos += 1;
        }
    }

    Ok(tensors)
}

fn skip_whitespace(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn parse_json_string(chars: &[char], pos: &mut usize) -> Result<String, String> {
    skip_whitespace(chars, pos);
    if *pos >= chars.len() || chars[*pos] != '"' {
        return Err(format!("expected '\"' at pos {}", *pos));
    }
    *pos += 1;
    let mut s = String::new();
    while *pos < chars.len() && chars[*pos] != '"' {
        if chars[*pos] == '\\' {
            *pos += 1;
            if *pos < chars.len() {
                s.push(chars[*pos]);
            }
        } else {
            s.push(chars[*pos]);
        }
        *pos += 1;
    }
    if *pos < chars.len() {
        *pos += 1; // skip closing '"'
    }
    Ok(s)
}

fn parse_json_number(chars: &[char], pos: &mut usize) -> Result<usize, String> {
    skip_whitespace(chars, pos);
    let start = *pos;
    while *pos < chars.len() && (chars[*pos].is_ascii_digit() || chars[*pos] == '-') {
        *pos += 1;
    }
    let s: String = chars[start..*pos].iter().collect();
    s.parse::<usize>()
        .map_err(|e| format!("invalid number '{}': {}", s, e))
}

fn skip_json_value(chars: &[char], pos: &mut usize) -> Result<(), String> {
    skip_whitespace(chars, pos);
    if *pos >= chars.len() {
        return Err("unexpected end of JSON".into());
    }
    match chars[*pos] {
        '"' => {
            parse_json_string(chars, pos)?;
        }
        '{' => {
            *pos += 1;
            let mut depth = 1;
            while *pos < chars.len() && depth > 0 {
                if chars[*pos] == '{' {
                    depth += 1;
                }
                if chars[*pos] == '}' {
                    depth -= 1;
                }
                *pos += 1;
            }
        }
        '[' => {
            *pos += 1;
            let mut depth = 1;
            while *pos < chars.len() && depth > 0 {
                if chars[*pos] == '[' {
                    depth += 1;
                }
                if chars[*pos] == ']' {
                    depth -= 1;
                }
                *pos += 1;
            }
        }
        _ => {
            while *pos < chars.len() && chars[*pos] != ',' && chars[*pos] != '}' {
                *pos += 1;
            }
        }
    }
    Ok(())
}

fn parse_tensor_info(chars: &[char], pos: &mut usize, name: &str) -> Result<TensorInfo, String> {
    skip_whitespace(chars, pos);
    if *pos >= chars.len() || chars[*pos] != '{' {
        return Err(format!("expected '{{' for tensor info of '{}'", name));
    }
    *pos += 1;

    let mut dtype = String::new();
    let mut shape = Vec::new();
    let mut data_start = 0usize;
    let mut data_end = 0usize;

    loop {
        skip_whitespace(chars, pos);
        if *pos >= chars.len() || chars[*pos] == '}' {
            *pos += 1;
            break;
        }

        let key = parse_json_string(chars, pos)?;
        skip_whitespace(chars, pos);
        if *pos < chars.len() && chars[*pos] == ':' {
            *pos += 1;
        }

        match key.as_str() {
            "dtype" => {
                dtype = parse_json_string(chars, pos)?;
            }
            "shape" => {
                skip_whitespace(chars, pos);
                if *pos < chars.len() && chars[*pos] == '[' {
                    *pos += 1;
                    loop {
                        skip_whitespace(chars, pos);
                        if *pos >= chars.len() || chars[*pos] == ']' {
                            *pos += 1;
                            break;
                        }
                        let n = parse_json_number(chars, pos)?;
                        shape.push(n);
                        skip_whitespace(chars, pos);
                        if *pos < chars.len() && chars[*pos] == ',' {
                            *pos += 1;
                        }
                    }
                }
            }
            "data_offsets" => {
                skip_whitespace(chars, pos);
                if *pos < chars.len() && chars[*pos] == '[' {
                    *pos += 1;
                    data_start = parse_json_number(chars, pos)?;
                    skip_whitespace(chars, pos);
                    if *pos < chars.len() && chars[*pos] == ',' {
                        *pos += 1;
                    }
                    data_end = parse_json_number(chars, pos)?;
                    skip_whitespace(chars, pos);
                    if *pos < chars.len() && chars[*pos] == ']' {
                        *pos += 1;
                    }
                }
            }
            _ => {
                skip_json_value(chars, pos)?;
            }
        }

        skip_whitespace(chars, pos);
        if *pos < chars.len() && chars[*pos] == ',' {
            *pos += 1;
        }
    }

    Ok(TensorInfo {
        name: name.to_string(),
        dtype,
        shape,
        data_start,
        data_end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header() {
        let header = r#"{"weight": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]}}"#;
        let tensors = parse_header(header).unwrap();
        assert_eq!(tensors.len(), 1);

        let w = &tensors["weight"];
        assert_eq!(w.dtype, "F32");
        assert_eq!(w.shape, vec![4, 3]);
        assert_eq!(w.data_start, 0);
        assert_eq!(w.data_end, 48);
    }

    #[test]
    fn test_parse_header_with_metadata() {
        let header = r#"{"__metadata__": {"format": "pt"}, "bias": {"dtype": "BF16", "shape": [10], "data_offsets": [0, 20]}}"#;
        let tensors = parse_header(header).unwrap();
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("bias"));
    }

    #[test]
    fn test_from_bytes_f32() {
        // Build a minimal safetensors file in memory
        let header = r#"{"x": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]}}"#;
        let header_bytes = header.as_bytes();
        let header_size = header_bytes.len() as u64;

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        // 3 floats: 1.0, 2.0, 3.0
        file_bytes.extend_from_slice(&1.0f32.to_le_bytes());
        file_bytes.extend_from_slice(&2.0f32.to_le_bytes());
        file_bytes.extend_from_slice(&3.0f32.to_le_bytes());

        let st = SafeTensorsFile::from_bytes(&file_bytes).unwrap();
        let data = st.get_tensor_f32("x").unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }
}
