//! Read the model format a `.dvm` was compiled to.
//!
//! A `.dvm` is a FlatBuffers buffer whose schema Kinara does not publish.
//! The converter records the model format as a one-field sub-table holding
//! a single byte. Converters built on the DVAPI 1.x SDKs write format 2 and
//! those built on the DVAPI 3.0 SDK write format 3; it is the major version
//! [`crate::Model::version`] reports once a model has loaded. The field's
//! id differs between the two
//! generations but its slot in the root table is at byte offset 80 in
//! both, so the reader locates it by that slot and validates every step on
//! the way. A buffer that fails any check is reported as unknown rather
//! than guessed at.
//!
//! Knowing the format before loading matters because the client library
//! reports a model built for the other runtime generation exactly as it
//! reports a damaged file.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{Abi, Error};

/// Offset of the format slot within the root table.
const FORMAT_SLOT: u64 = 80;
/// Largest root vtable accepted, in bytes.
const MAX_VTABLE: u16 = 1024;

/// Random-access byte source the reader runs over.
trait Source {
    fn len(&mut self) -> std::io::Result<u64>;
    /// Fills `buf` from `offset`. The caller has already checked the range
    /// against [`Source::len`].
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()>;
}

impl Source for &[u8] {
    fn len(&mut self) -> std::io::Result<u64> {
        Ok(<[u8]>::len(self) as u64)
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        let start = offset as usize;
        buf.copy_from_slice(&self[start..start + buf.len()]);
        Ok(())
    }
}

impl Source for File {
    fn len(&mut self) -> std::io::Result<u64> {
        Ok(self.metadata()?.len())
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.seek(SeekFrom::Start(offset))?;
        self.read_exact(buf)
    }
}

/// Bounds-checked little-endian reads. `Ok(None)` is out of bounds.
struct Reader<S> {
    src: S,
    len: u64,
}

impl<S: Source> Reader<S> {
    fn bytes<const N: usize>(&mut self, offset: u64) -> std::io::Result<Option<[u8; N]>> {
        match offset.checked_add(N as u64) {
            Some(end) if end <= self.len => {
                let mut buf = [0u8; N];
                self.src.read_at(offset, &mut buf)?;
                Ok(Some(buf))
            }
            _ => Ok(None),
        }
    }

    fn u16(&mut self, offset: u64) -> std::io::Result<Option<u16>> {
        Ok(self.bytes(offset)?.map(u16::from_le_bytes))
    }

    fn u32(&mut self, offset: u64) -> std::io::Result<Option<u32>> {
        Ok(self.bytes(offset)?.map(u32::from_le_bytes))
    }

    fn i32(&mut self, offset: u64) -> std::io::Result<Option<i32>> {
        Ok(self.bytes(offset)?.map(i32::from_le_bytes))
    }
}

/// Follows a FlatBuffers soffset from a table to its vtable.
fn vtable_of(table: u64, soffset: i32) -> Option<u64> {
    u64::try_from(i64::try_from(table).ok()? - i64::from(soffset)).ok()
}

/// Unwraps an `Option` inside the reader, turning `None` into "unknown".
macro_rules! need {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return Ok(None),
        }
    };
}

/// Fails validation unless the condition holds.
macro_rules! check {
    ($cond:expr) => {
        if !$cond {
            return Ok(None);
        }
    };
}

fn read_format<S: Source>(mut src: S) -> std::io::Result<Option<u8>> {
    let len = src.len()?;
    let mut r = Reader { src, len };

    let root = u64::from(need!(r.u32(0)?));
    check!(root >= 4 && root + FORMAT_SLOT + 4 <= len);

    let vt = need!(vtable_of(root, need!(r.i32(root)?)));
    check!(vt + 4 <= len);
    let vsize = need!(r.u16(vt)?);
    let tsize = need!(r.u16(vt + 2)?);
    check!(vsize.is_multiple_of(2) && (4..=MAX_VTABLE).contains(&vsize));
    check!(vt + u64::from(vsize) <= len);
    check!(u64::from(tsize) >= FORMAT_SLOT + 4);

    // The vtable usually sits near the end of the file, so read it in one go.
    let mut entries = [0u8; MAX_VTABLE as usize];
    let entries = &mut entries[..usize::from(vsize) - 4];
    r.src.read_at(vt + 4, entries)?;
    check!(
        entries
            .chunks_exact(2)
            .any(|e| u64::from(u16::from_le_bytes([e[0], e[1]])) == FORMAT_SLOT)
    );

    let slot = root + FORMAT_SLOT;
    let t = slot + u64::from(need!(r.u32(slot)?));
    check!(t + 5 <= len);

    let svt = need!(vtable_of(t, need!(r.i32(t)?)));
    check!(svt + 6 <= len);
    check!(need!(r.u16(svt)?) == 6);
    check!(need!(r.u16(svt + 2)?) >= 5);
    check!(need!(r.u16(svt + 4)?) == 4);

    let [format] = need!(r.bytes::<1>(t + 4)?);
    check!((1..=15).contains(&format));
    Ok(Some(format))
}

/// Reads the model format from the contents of a `.dvm`: 2 for DVAPI 1.x
/// runtimes, 3 for DVAPI 3.x.
///
/// Returns `None` when the buffer does not have the expected structure:
/// a model from a converter generation that lays the format out
/// differently, a truncated file, or something that is not a model.
///
/// ```
/// assert_eq!(ara2::read_model_format(b"not a model"), None);
/// ```
pub fn read_model_format(data: &[u8]) -> Option<u8> {
    read_format(data).ok().flatten()
}

/// Reads the model format from a `.dvm` file, without reading the whole
/// file. See [`read_model_format`].
///
/// `Err` only when the file cannot be opened or read; a file whose
/// structure is not recognised is `Ok(None)`.
pub fn read_model_format_from_file(path: &Path) -> Result<Option<u8>, Error> {
    let file = File::open(path)?;
    match read_format(file) {
        Ok(v) => Ok(v),
        // The length was checked first, so this is a file that shrank
        // under the reader, which is no structure at all.
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Whether a model's format suits the runtime generation it is loaded on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelCompatibility {
    /// The model was built for the other runtime generation and has to be
    /// reconverted with a converter built on this runtime's SDK.
    Mismatch {
        /// The model's format.
        model_format: u8,
        /// The runtime it was loaded on.
        runtime: Abi,
    },
    /// The model's format suits this runtime generation.
    Matching {
        /// The model's format.
        model_format: u8,
    },
    /// The format could not be read, or is one this build does not map to
    /// a runtime generation.
    Unknown,
}

impl ModelCompatibility {
    /// Compares a model format, as returned by [`read_model_format`], with
    /// the runtime's DVAPI generation.
    ///
    /// Format 2 belongs to DVAPI 1.x runtimes and format 3 to DVAPI 3.x;
    /// any other format is [`ModelCompatibility::Unknown`].
    ///
    /// ```
    /// use ara2::{Abi, ModelCompatibility};
    /// assert!(matches!(
    ///     ModelCompatibility::new(Some(2), Abi::V3_0),
    ///     ModelCompatibility::Mismatch { .. }
    /// ));
    /// ```
    pub fn new(format: Option<u8>, runtime: Abi) -> Self {
        let Some(model_format) = format else {
            return ModelCompatibility::Unknown;
        };
        match format_runtime_major(model_format) {
            Some(major) if major == runtime.major() => {
                ModelCompatibility::Matching { model_format }
            }
            Some(_) => ModelCompatibility::Mismatch {
                model_format,
                runtime,
            },
            None => ModelCompatibility::Unknown,
        }
    }

    /// The model format, when it was read.
    pub fn model_format(&self) -> Option<u8> {
        match self {
            ModelCompatibility::Mismatch { model_format, .. }
            | ModelCompatibility::Matching { model_format } => Some(*model_format),
            ModelCompatibility::Unknown => None,
        }
    }
}

/// The DVAPI major version of the runtimes a model format is built for.
pub(crate) fn format_runtime_major(format: u8) -> Option<u8> {
    match format {
        2 => Some(1),
        3 => Some(3),
        _ => None,
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;

    /// Builds a minimal FlatBuffers-shaped model: root offset, root table,
    /// a one-field sub-table holding the format byte and padding, and the root vtable at
    /// the end of the buffer as converters lay it out.
    struct Model {
        format: u8,
        slot_entries: Vec<u16>,
        sub_vtable: Vec<u16>,
    }

    impl Model {
        fn new(major: u8) -> Self {
            Model {
                format: major,
                slot_entries: vec![4, 8, 80],
                sub_vtable: vec![6, 5, 4],
            }
        }

        fn build(&self) -> Vec<u8> {
            // Root offset, then the root table: soffset and 84 bytes of fields.
            let root = 4usize;
            let mut b = (root as u32).to_le_bytes().to_vec();
            b.extend_from_slice(&[0u8; 88]);
            // The sub-table's vtable, then the sub-table pointing back at it.
            let svt = b.len();
            for v in &self.sub_vtable {
                b.extend_from_slice(&v.to_le_bytes());
            }
            while !b.len().is_multiple_of(4) {
                b.push(0);
            }
            let t = b.len();
            b.extend_from_slice(&((t - svt) as i32).to_le_bytes());
            b.extend_from_slice(&[self.format, 0, 0, 0]);
            let slot = root + 80;
            b[slot..slot + 4].copy_from_slice(&((t - slot) as u32).to_le_bytes());
            // The root vtable at the end.
            let vt = b.len();
            let vsize = 4 + 2 * self.slot_entries.len() as u16;
            b.extend_from_slice(&vsize.to_le_bytes());
            b.extend_from_slice(&88u16.to_le_bytes());
            for e in &self.slot_entries {
                b.extend_from_slice(&e.to_le_bytes());
            }
            b[root..root + 4].copy_from_slice(&((root as i64 - vt as i64) as i32).to_le_bytes());
            b
        }
    }

    #[test]
    fn offline_valid_formats_read() {
        assert_eq!(read_model_format(&Model::new(2).build()), Some(2));
        assert_eq!(read_model_format(&Model::new(3).build()), Some(3));
        // Only the format byte is declared; what follows it is padding.
        let mut b = Model::new(3).build();
        let format_at = b.len() - 10 - 4;
        b[format_at + 1..format_at + 4].copy_from_slice(&[7, 7, 7]);
        assert_eq!(read_model_format(&b), Some(3));
    }

    #[test]
    fn offline_no_slot_80_is_unknown() {
        let mut m = Model::new(3);
        m.slot_entries = vec![4, 8, 76];
        assert_eq!(read_model_format(&m.build()), None);
    }

    #[test]
    fn offline_sub_table_with_two_fields_is_unknown() {
        let mut m = Model::new(3);
        m.sub_vtable = vec![8, 12, 4, 8];
        assert_eq!(read_model_format(&m.build()), None);
    }

    #[test]
    fn offline_sub_table_field_elsewhere_is_unknown() {
        let mut m = Model::new(3);
        m.sub_vtable = vec![6, 8, 6];
        assert_eq!(read_model_format(&m.build()), None);
        m.sub_vtable = vec![6, 4, 4];
        assert_eq!(read_model_format(&m.build()), None);
    }

    #[test]
    fn offline_major_out_of_range_is_unknown() {
        assert_eq!(read_model_format(&Model::new(0).build()), None);
        assert_eq!(read_model_format(&Model::new(16).build()), None);
    }

    #[test]
    fn offline_out_of_bounds_root_is_unknown() {
        let mut b = Model::new(3).build();
        let len = b.len() as u32;
        b[0..4].copy_from_slice(&len.to_le_bytes());
        assert_eq!(read_model_format(&b), None);
        b[0..4].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(read_model_format(&b), None);
    }

    #[test]
    fn offline_out_of_bounds_vtable_is_unknown() {
        let mut b = Model::new(3).build();
        let root = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let past_end = -(b.len() as i32);
        b[root..root + 4].copy_from_slice(&past_end.to_le_bytes());
        assert_eq!(read_model_format(&b), None);
        // A soffset pointing before the start of the buffer.
        b[root..root + 4].copy_from_slice(&i32::MAX.to_le_bytes());
        assert_eq!(read_model_format(&b), None);
    }

    #[test]
    fn offline_oversized_vtable_is_unknown() {
        let mut b = Model::new(3).build();
        let vt = b.len() - 10;
        b[vt..vt + 2].copy_from_slice(&2048u16.to_le_bytes());
        assert_eq!(read_model_format(&b), None);
        b[vt..vt + 2].copy_from_slice(&9u16.to_le_bytes());
        assert_eq!(read_model_format(&b), None);
    }

    #[test]
    fn offline_out_of_bounds_uoffset_is_unknown() {
        let mut b = Model::new(3).build();
        let root = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let slot = root + 80;
        b[slot..slot + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(read_model_format(&b), None);
    }

    #[test]
    fn offline_truncated_is_unknown() {
        let b = Model::new(3).build();
        for cut in [0, 3, 4, 50, b.len() - 1] {
            assert_eq!(read_model_format(&b[..cut]), None, "cut at {cut}");
        }
    }

    #[test]
    fn offline_file_reader_matches_buffer_reader() {
        let dir = std::env::temp_dir().join(format!("ara2-dvm-format-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.dvm");
        std::fs::write(&path, Model::new(2).build()).unwrap();
        assert_eq!(read_model_format_from_file(&path).unwrap(), Some(2));
        std::fs::write(&path, b"short").unwrap();
        assert_eq!(read_model_format_from_file(&path).unwrap(), None);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(matches!(
            read_model_format_from_file(&path),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn offline_compatibility_follows_the_format_major() {
        use ModelCompatibility::*;
        assert_eq!(
            ModelCompatibility::new(Some(2), Abi::V3_0),
            Mismatch {
                model_format: 2,
                runtime: Abi::V3_0
            }
        );
        assert_eq!(
            ModelCompatibility::new(Some(3), Abi::V1_3),
            Mismatch {
                model_format: 3,
                runtime: Abi::V1_3
            }
        );
        assert_eq!(
            ModelCompatibility::new(Some(2), Abi::V1_1),
            Matching { model_format: 2 }
        );
        assert_eq!(
            ModelCompatibility::new(Some(2), Abi::V1_3),
            Matching { model_format: 2 }
        );
        assert_eq!(
            ModelCompatibility::new(Some(3), Abi::V3_0),
            Matching { model_format: 3 }
        );
        assert_eq!(ModelCompatibility::new(Some(4), Abi::V3_0), Unknown);
        assert_eq!(ModelCompatibility::new(None, Abi::V3_0), Unknown);
    }

    /// Runs the reader over real models named by `ARA2_TEST_FORMAT_FILES`,
    /// a comma-separated list of `path=format` pairs (`none` for unknown).
    #[test]
    fn offline_real_files_when_given() {
        let Ok(list) = std::env::var("ARA2_TEST_FORMAT_FILES") else {
            return;
        };
        for pair in list.split(',').filter(|p| !p.is_empty()) {
            let (path, want) = pair.rsplit_once('=').expect("path=format");
            let got = read_model_format_from_file(Path::new(path)).unwrap();
            let got = got.map_or("none".to_owned(), |v| v.to_string());
            eprintln!("{path}: {got}");
            assert_eq!(got, want, "{path}");
        }
    }
}
