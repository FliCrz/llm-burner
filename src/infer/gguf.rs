//! Hand-rolled GGUF reader for the Vulkan inference engine.
//!
//! Lowest possible dependency footprint: the parser mmap's the model file and
//! walks the header, metadata key/value table, and tensor table directly over
//! the mapped `&[u8]` — no `rlx-gguf`, no `rayon`, no error/result wrappers
//! beyond `anyhow`. Weight bytes stay in the mapping and are copied into a
//! unified GPU buffer at load ([`crate::infer::vk`]).
//!
//! ## Format (v2/v3, matching `rlx_gguf`'s reader)
//!
//! ```text
//! u32  magic        "GGUF" (0x4655_4747)
//! u32  version      2 or 3
//! u64  tensor_count
//! u64  kv_count
//! kv_count × (u64 len, key utf8, value)
//! tensor_count × (u64 len, name, u32 n_dims, n_dims × u64 dim, u32 ggml_type, u64 offset)
//! pad to `general.alignment` (default 32)
//! tensor data ...                                     (start = `data_offset`)
//! ```
//!
//! Tensor `shape` is kept in ggml order (innermost/last dimension first) and
//! `offset` is relative to `data_offset`, exactly like the reference reader.
//! Metadata values use the `GGUFValueType` tags: `{U8,I8,U16,I16,U32,I32,F32,
//! Bool,String}=0..8`, `Array=9`, `{U64,I64,F64}=10..12`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// `"GGUF"` little-endian.
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Assumed `general.alignment` when the metadata omits it.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// On-disk ggml element type tags (subset of llama.cpp's `ggml_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
    I8,
    I16,
    I32,
    I64,
    F64,
}

impl GgmlDType {
    /// Map an on-disk `ggml_type` tag; unknown tags error out.
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::BF16,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            other => bail!("unknown ggml type tag {other}"),
        })
    }

    /// Bytes per scalar element (0 for packed block quantizations).
    pub fn scalar_bytes(self) -> Option<u64> {
        match self {
            Self::F32 | Self::I32 => Some(4),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::F64 | Self::I64 => Some(8),
            Self::I8 => Some(1),
            _ => None,
        }
    }

    /// True for the K-quant (super-block) family used by `llm-burner` exports.
    pub fn is_k_quant(self) -> bool {
        matches!(
            self,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K
        )
    }
}

/// On-disk bytes per element for the packed quantizations (fractional — e.g.
/// Q4_0 stores 32 values in 18 bytes). Only valid on a type whose element
/// count is divisible by its block size.
pub fn packed_bytes_per_elem(dt: GgmlDType) -> Option<f64> {
    match dt {
        GgmlDType::Q4_0 => Some(18.0 / 32.0),
        GgmlDType::Q4_1 => Some(20.0 / 32.0),
        GgmlDType::Q5_0 => Some(22.0 / 32.0),
        GgmlDType::Q5_1 => Some(24.0 / 32.0),
        GgmlDType::Q8_0 => Some(34.0 / 32.0),
        GgmlDType::Q2K => Some(84.0 / 256.0),
        GgmlDType::Q3K => Some(110.0 / 256.0),
        GgmlDType::Q4K => Some(144.0 / 256.0),
        GgmlDType::Q5K => Some(176.0 / 256.0),
        GgmlDType::Q6K => Some(210.0 / 256.0),
        GgmlDType::Q8K => Some(260.0 / 256.0),
        _ => None,
    }
}

/// One tensor described by the header's tensor table.
#[derive(Debug, Clone)]
pub struct GgufTensor {
    /// Storage name, e.g. `blk.0.attn_q.weight`.
    pub name: String,
    /// Shape in ggml order (innermost/last dim first), e.g. `[in, out]`.
    pub shape: Vec<u64>,
    pub dtype: GgmlDType,
    /// Byte offset within the tensor data segment (relative to `data_offset`).
    pub offset: u64,
}

impl GgufTensor {
    /// Product of the shape dims.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Number of bytes this tensor occupies in the data segment.
    pub fn byte_len(&self) -> u64 {
        match self.dtype.scalar_bytes() {
            Some(b) => self.n_elements() * b,
            None => {
                let per_elem = packed_bytes_per_elem(self.dtype).unwrap();
                (self.n_elements() as f64 * per_elem).ceil() as u64
            }
        }
    }
}

/// A parsed GGUF header (metadata + tensor table). The weight bytes live in a
/// [`GgufFile`] so this owns no backing storage.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, GgufMeta>,
    pub tensors: HashMap<String, GgufTensor>,
    /// Byte offset of the tensor data segment in the file.
    pub data_offset: u64,
}

/// A parsed metadata key/value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMeta {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    String(String),
    Array(Vec<GgufMeta>),
}

impl GgufMeta {
    /// Interpret as `u64` (mirrors `rlx_gguf::MetaValue::as_u64`).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// First element of an Array, interpreted as `u32`; also accepts a scalar.
    pub fn as_first_u32(&self) -> Option<u32> {
        match self {
            Self::Array(items) => items.first().and_then(GgufMeta::as_u32),
            other => other.as_u32(),
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            Self::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            _ => None,
        }
    }
}

impl GgufHeader {
    /// Parse the header from complete file bytes.
    pub fn parse(map: &[u8]) -> Result<Self> {
        let mut p = Cursor::new(map);
        let magic = p.u32()?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (magic {magic:#x})");
        }
        let version = p.u32()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version} (expected 2 or 3)");
        }

        let tensor_count = p.u64()?;
        let kv_count = p.u64()?;

        // Metadata. `kv_count` is untrusted header data; cap the pre-allocation
        // hint so a hostile length can't trigger a giant up-front allocation.
        let mut metadata = HashMap::with_capacity((kv_count as usize).min(1 << 16));
        for _ in 0..kv_count {
            let key = p.string(version)?;
            let value = p.value(version)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufMeta::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);

        let mut tensors = HashMap::with_capacity((tensor_count as usize).min(1 << 16));
        for _ in 0..tensor_count {
            let name = p.string(version)?;
            let n_dims = p.u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims.min(1 << 8));
            for _ in 0..n_dims {
                shape.push(if version == 2 { p.u64()? } else { p.u64()? });
            }
            let dtype = GgmlDType::from_u32(p.u32()?).with_context(|| format!("tensor `{name}`"))?;
            let offset = p.u64()?;
            tensors.insert(
                name.clone(),
                GgufTensor {
                    name,
                    shape,
                    dtype,
                    offset,
                },
            );
        }

        // Data segment starts at the next alignment boundary after the header.
        let pos = p.pos();
        let pad = (alignment - (pos as u64 % alignment)) % alignment;
        let data_offset = pos as u64 + pad;

        if data_offset as usize > map.len() {
            bail!(
                "GGUF data offset {data_offset} past mapped length {}",
                map.len()
            );
        }

        Ok(Self {
            version,
            alignment,
            metadata,
            tensors,
            data_offset,
        })
    }
}

/// A GGUF file bound to its weight bytes: a parsed [`GgufHeader`] plus either
/// an mmap of the file or (for tests / in-memory blobs) an owned buffer.
pub struct GgufFile {
    pub header: GgufHeader,
    data: GgufData,
}

enum GgufData {
    Owned(Vec<u8>),
    Mmap(memmap2::Mmap),
}

impl std::ops::Deref for GgufData {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            GgufData::Owned(v) => v.as_slice(),
            GgufData::Mmap(m) => m.as_ref(),
        }
    }
}

impl GgufFile {
    /// Open `path`, mmap it, and parse its header.
    pub fn from_path(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("open `{}`", path.display()))?;
        // SAFETY: read-only mmap of a file we keep a handle to; the mapping is
        // never mutated and pages are refaulted from disk on access.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmap `{}`", path.display()))?;
        let header = GgufHeader::parse(&map)?;
        Ok(Self {
            header,
            data: GgufData::Mmap(map),
        })
    }

    /// Wrap an in-memory blob (used by tests and small fixtures).
    pub fn from_owned(bytes: Vec<u8>) -> Result<Self> {
        let header = GgufHeader::parse(&bytes)?;
        Ok(Self {
            header,
            data: GgufData::Owned(bytes),
        })
    }

    /// The tensor's data bytes within this file's backing storage.
    pub fn tensor_bytes(&self, t: &GgufTensor) -> Result<&[u8]> {
        let start = self.header.data_offset as usize + t.offset as usize;
        let len = t.byte_len() as usize;
        let end = start
            .checked_add(len)
            .context("tensor byte range overflow")?;
        self.data
            .get(start..end)
            .with_context(|| format!("tensor `{}` byte range out of bounds", t.name))
    }

    pub fn get(&self, name: &str) -> Option<&GgufTensor> {
        self.header.tensors.get(name)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.header.tensors.keys().map(String::as_str)
    }
}

/// Bounds-checked, little-endian reader over a byte slice.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .with_context(|| format!("cursor overflow at {}+{n}", self.pos))?;
        let out = self
            .bytes
            .get(self.pos..end)
            .with_context(|| format!("truncated GGUF: need {n} bytes at {}", self.pos))?;
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self, _version: u32) -> Result<String> {
        let len = self.u64()? as usize;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).context("non-UTF8 GGUF string")
    }

    fn value(&mut self, version: u32) -> Result<GgufMeta> {
        let ty = self.u32()?;
        Ok(match ty {
            0 => GgufMeta::U8(self.u8()?),
            1 => GgufMeta::I8(self.i8()?),
            2 => GgufMeta::U16(self.u16()?),
            3 => GgufMeta::I16(self.i16()?),
            4 => GgufMeta::U32(self.u32()?),
            5 => GgufMeta::I32(self.i32()?),
            6 => GgufMeta::F32(self.f32()?),
            7 => GgufMeta::Bool(self.u8()? != 0),
            8 => GgufMeta::String(self.string(version)?),
            9 => {
                let inner = self.u32()?;
                let len = if version == 2 { self.u64()? as usize } else { self.u64()? as usize };
                let mut out = Vec::with_capacity(len.min(1 << 16));
                for _ in 0..len {
                    out.push(self.scalar(inner, version)?);
                }
                GgufMeta::Array(out)
            }
            10 => GgufMeta::U64(self.u64()?),
            11 => GgufMeta::I64(self.i64()?),
            12 => GgufMeta::F64(self.f64()?),
            other => bail!("unknown metadata value type {other}"),
        })
    }

    fn scalar(&mut self, ty: u32, version: u32) -> Result<GgufMeta> {
        Ok(match ty {
            0 => GgufMeta::U8(self.u8()?),
            1 => GgufMeta::I8(self.i8()?),
            2 => GgufMeta::U16(self.u16()?),
            3 => GgufMeta::I16(self.i16()?),
            4 => GgufMeta::U32(self.u32()?),
            5 => GgufMeta::I32(self.i32()?),
            6 => GgufMeta::F32(self.f32()?),
            7 => GgufMeta::Bool(self.u8()? != 0),
            8 => GgufMeta::String(self.string(version)?),
            10 => GgufMeta::U64(self.u64()?),
            11 => GgufMeta::I64(self.i64()?),
            12 => GgufMeta::F64(self.f64()?),
            9 => bail!("nested arrays not allowed in GGUF metadata"),
            other => bail!("unknown array element type {other}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_str<'a>(h: &'a GgufHeader, key: &str) -> Option<&'a str> {
        h.metadata.get(key).and_then(GgufMeta::as_str)
    }

    fn meta_u64(h: &GgufHeader, key: &str) -> Option<u64> {
        h.metadata.get(key).and_then(GgufMeta::as_u64)
    }

    /// Build a minimal v3 GGUF in memory (architecture + one F32 tensor).
    fn build_v3_gguf(alignment: u64) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend((GGUF_MAGIC as u32).to_le_bytes());
        v.extend(3u32.to_le_bytes());
        // tensor_count, kv_count
        v.extend(1u64.to_le_bytes());
        v.extend(3u64.to_le_bytes());
        // kv: general.architecture = "llama"
        let k = "general.architecture".as_bytes();
        v.extend((k.len() as u64).to_le_bytes());
        v.extend(k);
        v.extend(8u32.to_le_bytes()); // string
        let val = "llama".as_bytes();
        v.extend((val.len() as u64).to_le_bytes());
        v.extend(val);
        // kv: block_count = 2
        let k = "block_count".as_bytes();
        v.extend((k.len() as u64).to_le_bytes());
        v.extend(k);
        v.extend(10u32.to_le_bytes()); // u64
        v.extend(2u64.to_le_bytes());
        // kv: general.alignment = alignment
        let k = "general.alignment".as_bytes();
        v.extend((k.len() as u64).to_le_bytes());
        v.extend(k);
        v.extend(10u32.to_le_bytes());
        v.extend(alignment.to_le_bytes());
        // tensor: "tok.weight", shape [8, 4] (ggml [w, h]), F32, offset 0
        let n = "tok.weight".as_bytes();
        v.extend((n.len() as u64).to_le_bytes());
        v.extend(n);
        v.extend(2u32.to_le_bytes()); // n_dims
        v.extend(8u64.to_le_bytes()); // dim0 (innermost)
        v.extend(4u64.to_le_bytes()); // dim1
        v.extend(0u32.to_le_bytes()); // F32
        v.extend(0u64.to_le_bytes()); // offset
        // pad to alignment
        while v.len() % alignment as usize != 0 {
            v.push(0);
        }
        // data: 8*4 f32 values 1..32
        for i in 1u32..=32 {
            v.extend((i as f32).to_le_bytes());
        }
        v
    }

    #[test]
    fn parses_v3_header_metadata_and_tensor() {
        let bytes = build_v3_gguf(32);
        let h = GgufHeader::parse(&bytes).unwrap();
        assert_eq!(h.version, 3);
        assert_eq!(meta_str(&h, "general.architecture"), Some("llama"));
        assert_eq!(meta_u64(&h, "block_count"), Some(2));
        assert_eq!(h.alignment, 32);
        let t = h.tensors.get("tok.weight").unwrap();
        assert_eq!(t.shape, vec![8, 4]);
        assert_eq!(t.dtype, GgmlDType::F32);
        assert_eq!(t.n_elements(), 32);
        assert_eq!(t.byte_len(), 128);
        let f = GgufFile::from_owned(bytes.clone()).unwrap();
        assert_eq!(f.tensor_bytes(t).unwrap().len(), 128);
        // data offset is 32-aligned
        assert_eq!(h.data_offset % 32, 0);
    }

    #[test]
    fn data_offset_respects_alignment() {
        // Build with a weird alignment so the pad math matters.
        let bytes = build_v3_gguf(64);
        let h = GgufHeader::parse(&bytes).unwrap();
        assert_eq!(h.alignment, 64);
        assert_eq!(h.data_offset % 64, 0);
        let f = GgufFile::from_owned(bytes).unwrap();
        let t = f.get("tok.weight").unwrap();
        let raw = f.tensor_bytes(t).unwrap();
        assert_eq!(raw.len(), 128);
        let first: f32 = f32::from_le_bytes(raw[0..4].try_into().unwrap());
        assert_eq!(first, 1.0);
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut b = build_v3_gguf(32);
        b[0] ^= 0xff;
        assert!(GgufHeader::parse(&b).is_err());
    }

    #[test]
    fn rejects_unknown_version() {
        let mut b = build_v3_gguf(32);
        b[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(GgufHeader::parse(&b).is_err());
    }

    #[test]
    fn rejects_unknown_dtype() {
        let mut b = build_v3_gguf(32);
        // Locate the dtype field by walking; simpler: overwrite the last u32
        // before the pad, which is the dtype for a single tensor file.
        // Compute its position: header prologue length is variable; search for
        // "tok.weight" then the dtype follows name+len+n_dims(4)+2 dims(16).
        let corpus = b.clone();
        let name_pos = corpus.windows(10).position(|w| w == b"tok.weight").unwrap();
        let pos_after_name = name_pos + 10;
        // name: (len u64) then bytes; then n_dims u32; then 2 dims u64+u64; then dtype
        let dtype_pos = pos_after_name + 4 + 16;
        let dt = u32::from_le_bytes(corpus[dtype_pos..dtype_pos + 4].try_into().unwrap());
        assert_eq!(dt, 0); // F32
        b[dtype_pos..dtype_pos + 4].copy_from_slice(&999u32.to_le_bytes());
        assert!(GgufHeader::parse(&b).is_err());
    }

    #[test]
    fn as_first_u32_reads_array_first_element() {
        let arr = GgufMeta::Array(vec![GgufMeta::U32(7), GgufMeta::U32(9)]);
        assert_eq!(arr.as_first_u32(), Some(7));
        let scalar = GgufMeta::U32(7);
        assert_eq!(scalar.as_first_u32(), Some(7));
        assert_eq!(scalar.as_u64(), Some(7));
        assert_eq!(GgufMeta::I64(-1).as_u64(), None);
        assert_eq!(GgufMeta::F32(1.5).as_f32(), Some(1.5));
    }

    #[test]
    fn dtype_tag_mapping_roundtrip() {
        let map: [(u32, GgmlDType); 18] = [
            (0, GgmlDType::F32),
            (1, GgmlDType::F16),
            (2, GgmlDType::Q4_0),
            (3, GgmlDType::Q4_1),
            (6, GgmlDType::Q5_0),
            (7, GgmlDType::Q5_1),
            (8, GgmlDType::Q8_0),
            (10, GgmlDType::Q2K),
            (11, GgmlDType::Q3K),
            (12, GgmlDType::Q4K),
            (13, GgmlDType::Q5K),
            (14, GgmlDType::Q6K),
            (15, GgmlDType::Q8K),
            (30, GgmlDType::BF16),
            (24, GgmlDType::I8),
            (25, GgmlDType::I16),
            (26, GgmlDType::I32),
            (28, GgmlDType::F64),
        ];
        for (tag, want) in map {
            assert_eq!(GgmlDType::from_u32(tag).unwrap(), want);
        }
        assert!(GgmlDType::from_u32(999).is_err());
    }

    #[test]
    fn k_quant_flag() {
        assert!(GgmlDType::Q4K.is_k_quant());
        assert!(!GgmlDType::F32.is_k_quant());
    }

    #[test]
    fn scalar_byte_counts() {
        assert_eq!(GgmlDType::F32.scalar_bytes(), Some(4));
        assert_eq!(GgmlDType::F16.scalar_bytes(), Some(2));
        assert_eq!(GgmlDType::BF16.scalar_bytes(), Some(2));
        assert_eq!(GgmlDType::I8.scalar_bytes(), Some(1));
        assert_eq!(GgmlDType::Q4K.scalar_bytes(), None);
    }

    #[test]
    fn tensor_offset_is_relative_to_data_segment() {
        let mut b = build_v3_gguf(32);
        // Bump the tensor's stored offset to 16 and rebuild a valid file image
        // (header terminates after `offset + 8`; pad; then 128 bytes of data).
        let name_pos = b.windows(10).position(|w| w == b"tok.weight").unwrap();
        let after_name = name_pos + 10;
        let offset_pos = after_name + 4 + 16 + 4; // n_dims + 2 dims + dtype
        b.truncate(offset_pos);
        b.extend_from_slice(&16u64.to_le_bytes());
        while b.len() % 32 != 0 {
            b.push(0);
        }
        // The tensor now sits 16 bytes into the data segment: pad out first.
        b.extend_from_slice(&[0u8; 16]);
        for i in 1u32..=32 {
            b.extend((i as f32).to_le_bytes());
        }
        let h = GgufHeader::parse(&b).unwrap();
        let f = GgufFile::from_owned(b).unwrap();
        let t = f.get("tok.weight").unwrap();
        assert_eq!(t.offset, 16);
        // data starts at data_offset; tensor at data_offset+16.
        assert_eq!(f.tensor_bytes(t).unwrap().len(), 128);
        assert!(h.data_offset >= 16);
    }
}