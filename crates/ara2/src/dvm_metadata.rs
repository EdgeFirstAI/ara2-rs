//! Read EdgeFirst metadata from compiled DVM models.
//!
//! DVM files may have a ZIP archive appended to the end containing
//! `edgefirst.json` and `labels.txt`. This is the same approach used
//! by TFLite metadata.
//!
//! The DVM loader reads from the front of the file while ZIP reads
//! from the back (End of Central Directory), so both coexist without
//! interference.

use crate::Error;
use serde::Deserialize;
use std::io::{Cursor, Read as _};
use zip::ZipArchive;

/// Filename for the EdgeFirst metadata JSON.
pub const METADATA_FILENAME: &str = "edgefirst.json";

/// Filename for the class labels.
pub const LABELS_FILENAME: &str = "labels.txt";

/// EdgeFirst metadata embedded in a DVM file.
///
/// This struct captures the commonly-used fields. The full metadata
/// is available via [`read_metadata_raw`] for cases where additional
/// fields are needed.
#[derive(Debug, Clone, Deserialize)]
pub struct DvmMetadata {
    /// Dataset information
    #[serde(default)]
    pub dataset: Option<DatasetInfo>,

    /// Input specification
    #[serde(default)]
    pub input: Option<InputSpec>,

    /// Model information (task type, version, etc.)
    #[serde(default)]
    pub model: Option<ModelInfo>,

    /// Deployment information (name, author, description)
    #[serde(default)]
    pub deployment: Option<DeploymentInfo>,

    /// Compilation information (present in DVM, not in source ONNX)
    #[serde(default)]
    pub compilation: Option<CompilationInfo>,

    /// Decoder version (e.g., "yolov8")
    #[serde(default)]
    pub decoder_version: Option<String>,

    /// NMS type (e.g., "class_agnostic")
    #[serde(default)]
    pub nms: Option<String>,

    /// Output specifications
    #[serde(default)]
    pub outputs: Vec<OutputSpec>,
}

impl DvmMetadata {
    /// Get the model task type (detect, segment, classify, pose).
    pub fn task(&self) -> Option<&str> {
        self.model.as_ref()?.model_task.as_deref()
    }

    /// Get the class labels from the dataset.
    pub fn classes(&self) -> &[String] {
        self.dataset
            .as_ref()
            .map(|d| d.classes.as_slice())
            .unwrap_or(&[])
    }
}

/// Dataset information from the metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct DatasetInfo {
    /// Class names for the model
    #[serde(default)]
    pub classes: Vec<String>,

    /// Dataset ID
    #[serde(default)]
    pub id: Option<String>,

    /// Dataset name
    #[serde(default)]
    pub name: Option<String>,
}

/// Model information from the metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    /// Model task type (detect, segment, classify, pose)
    #[serde(default)]
    pub model_task: Option<String>,

    /// Model size variant (n, s, m, l, x)
    #[serde(default)]
    pub model_size: Option<String>,

    /// Model version string
    #[serde(default)]
    pub model_version: Option<String>,

    /// Whether detection is enabled
    #[serde(default)]
    pub detection: bool,

    /// Whether segmentation is enabled
    #[serde(default)]
    pub segmentation: bool,
}

/// Deployment information from the metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct DeploymentInfo {
    /// Model name for deployment
    #[serde(default)]
    pub model_name: Option<String>,

    /// Human-readable name
    #[serde(default)]
    pub name: Option<String>,

    /// Author or organization
    #[serde(default)]
    pub author: Option<String>,

    /// Description
    #[serde(default)]
    pub description: Option<String>,
}

/// Output tensor specification from the metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputSpec {
    /// Output index
    #[serde(default)]
    pub index: Option<u32>,

    /// Output name
    #[serde(default)]
    pub name: Option<String>,

    /// Output type (detection, segmentation, etc.)
    #[serde(default, rename = "type")]
    pub output_type: Option<String>,

    /// Decoder to use
    #[serde(default)]
    pub decoder: Option<String>,

    /// Whether to decode this output
    #[serde(default)]
    pub decode: bool,

    /// Data type (float32, int8, etc.)
    #[serde(default)]
    pub dtype: Option<String>,

    /// Tensor shape
    #[serde(default)]
    pub shape: Vec<i64>,
}

/// Input specification from the metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct InputSpec {
    /// Size string (e.g., "640x480")
    #[serde(default)]
    pub size: Option<String>,

    /// Number of input channels
    #[serde(default)]
    pub input_channels: Option<u32>,

    /// Number of output channels
    #[serde(default)]
    pub output_channels: Option<u32>,

    /// Camera adaptor type (rgb, bgr, etc.)
    #[serde(default)]
    pub cameraadaptor: Option<String>,
}

impl InputSpec {
    /// Parse the size string into (width, height).
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        let size = self.size.as_ref()?;
        let parts: Vec<&str> = size.split('x').collect();
        if parts.len() == 2 {
            let w = parts[0].parse().ok()?;
            let h = parts[1].parse().ok()?;
            Some((w, h))
        } else {
            None
        }
    }
}

/// Compilation information added when creating the DVM.
#[derive(Debug, Clone, Deserialize)]
pub struct CompilationInfo {
    /// Target hardware (e.g., "ara-2")
    #[serde(default)]
    pub target: Option<String>,

    /// Model format (e.g., "dvm")
    #[serde(default)]
    pub format: Option<String>,

    /// Performance/power/area metrics
    #[serde(default)]
    pub ppa: Option<PpaMetrics>,
}

/// Performance, power, and area metrics from compilation.
#[derive(Debug, Clone, Deserialize)]
pub struct PpaMetrics {
    /// Inferences per second
    #[serde(default)]
    pub ips: Option<f64>,

    /// Power consumption in milliwatts
    #[serde(default)]
    pub power_mw: Option<f64>,

    /// Execution cycles
    #[serde(default)]
    pub cycles: Option<u64>,

    /// DDR bandwidth in MB/s
    #[serde(default)]
    pub ddr_bw_mbps: Option<f64>,
}

/// Read EdgeFirst metadata from a DVM file.
///
/// Parses the `edgefirst.json` from the ZIP archive appended to the
/// DVM file into a [`DvmMetadata`] struct.
///
/// # Arguments
/// * `data` - The complete DVM file contents
///
/// # Returns
/// * `Ok(Some(metadata))` - Metadata was found and parsed
/// * `Ok(None)` - No ZIP archive or no metadata file present
/// * `Err(_)` - ZIP or JSON parsing error
pub fn read_metadata(data: &[u8]) -> Result<Option<DvmMetadata>, Error> {
    let json = match read_metadata_raw(data)? {
        Some(j) => j,
        None => return Ok(None),
    };

    let metadata: DvmMetadata = serde_json::from_str(&json)?;
    Ok(Some(metadata))
}

/// Read the raw EdgeFirst metadata JSON from a DVM file.
///
/// Returns the unparsed JSON string for cases where access to
/// additional fields is needed beyond what [`DvmMetadata`] captures.
///
/// # Arguments
/// * `data` - The complete DVM file contents
///
/// # Returns
/// * `Ok(Some(json))` - Raw JSON string from `edgefirst.json`
/// * `Ok(None)` - No ZIP archive or no metadata file present
/// * `Err(_)` - ZIP reading error
pub fn read_metadata_raw(data: &[u8]) -> Result<Option<String>, Error> {
    let cursor = Cursor::new(data);

    // Try to open as ZIP - if this fails, there's no metadata
    let mut archive = match ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(zip::result::ZipError::InvalidArchive(_)) => return Ok(None),
        Err(e) => return Err(Error::Zip(e)),
    };

    // Look for edgefirst.json
    let mut file = match archive.by_name(METADATA_FILENAME) {
        Ok(f) => f,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(e) => return Err(Error::Zip(e)),
    };

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(Some(contents))
}

/// Read class labels from a DVM file.
///
/// Reads the `labels.txt` file from the ZIP archive. Each line is
/// one class label.
///
/// # Arguments
/// * `data` - The complete DVM file contents
///
/// # Returns
/// * `Ok(labels)` - Vector of class label strings (empty if not present)
/// * `Err(_)` - ZIP reading error
pub fn read_labels(data: &[u8]) -> Result<Vec<String>, Error> {
    let cursor = Cursor::new(data);

    // Try to open as ZIP
    let mut archive = match ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(zip::result::ZipError::InvalidArchive(_)) => return Ok(vec![]),
        Err(e) => return Err(Error::Zip(e)),
    };

    // Look for labels.txt
    let mut file = match archive.by_name(LABELS_FILENAME) {
        Ok(f) => f,
        Err(zip::result::ZipError::FileNotFound) => return Ok(vec![]),
        Err(e) => return Err(Error::Zip(e)),
    };

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let labels: Vec<String> = contents
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();

    Ok(labels)
}

/// Check if a DVM file has embedded metadata.
///
/// A fast check that doesn't parse the metadata content.
///
/// # Arguments
/// * `data` - The complete DVM file contents
///
/// # Returns
/// `true` if the file contains a ZIP archive with `edgefirst.json`
pub fn has_metadata(data: &[u8]) -> bool {
    let cursor = Cursor::new(data);

    let archive = match ZipArchive::new(cursor) {
        Ok(a) => a,
        Err(_) => return false,
    };

    archive.file_names().any(|n| n == METADATA_FILENAME)
}

/// Read metadata from a DVM file path.
///
/// Convenience function that reads the file and parses metadata.
///
/// # Arguments
/// * `path` - Path to the DVM file
pub fn read_metadata_from_file(path: &std::path::Path) -> Result<Option<DvmMetadata>, Error> {
    let data = std::fs::read(path)?;
    read_metadata(&data)
}

/// Read labels from a DVM file path.
///
/// Convenience function that reads the file and extracts labels.
///
/// # Arguments
/// * `path` - Path to the DVM file
pub fn read_labels_from_file(path: &std::path::Path) -> Result<Vec<String>, Error> {
    let data = std::fs::read(path)?;
    read_labels(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_metadata_returns_none() {
        // Raw bytes that aren't a ZIP
        let data = b"not a zip file";
        assert!(read_metadata(data).unwrap().is_none());
        assert!(read_labels(data).unwrap().is_empty());
        assert!(!has_metadata(data));
    }

    use std::io::Write as _;

    /// Create a synthetic DVM file with embedded ZIP metadata.
    fn make_dvm_with_metadata(json: &str, labels: Option<&str>) -> Vec<u8> {
        let mut data = b"FAKE_DVM_HEADER_DATA".to_vec();
        let cursor = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(cursor);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);

        zip.start_file(METADATA_FILENAME, options).unwrap();
        zip.write_all(json.as_bytes()).unwrap();

        if let Some(labels_content) = labels {
            zip.start_file(LABELS_FILENAME, options).unwrap();
            zip.write_all(labels_content.as_bytes()).unwrap();
        }

        let cursor = zip.finish().unwrap();
        data.extend_from_slice(&cursor.into_inner());
        data
    }

    #[test]
    fn test_metadata_roundtrip() {
        let json = r#"{
            "model": {"model_task": "detect", "model_size": "s", "detection": true},
            "dataset": {"classes": ["person", "car", "bike"]},
            "input": {"size": "640x480", "input_channels": 3},
            "deployment": {"model_name": "yolov8s", "author": "EdgeFirst"},
            "decoder_version": "yolov8"
        }"#;

        let data = make_dvm_with_metadata(json, None);
        let meta = read_metadata(&data)
            .unwrap()
            .expect("metadata should be present");

        assert_eq!(meta.task(), Some("detect"));
        assert_eq!(meta.classes(), &["person", "car", "bike"]);
        assert_eq!(meta.decoder_version.as_deref(), Some("yolov8"));
        assert_eq!(meta.input.as_ref().unwrap().dimensions(), Some((640, 480)));
        assert_eq!(
            meta.deployment.as_ref().unwrap().author.as_deref(),
            Some("EdgeFirst")
        );
    }

    #[test]
    fn test_labels_roundtrip() {
        let json = r#"{"model": {}}"#;
        let labels_text = "person\ncar\nbike\n";
        let data = make_dvm_with_metadata(json, Some(labels_text));

        let labels = read_labels(&data).unwrap();
        assert_eq!(labels, vec!["person", "car", "bike"]);
    }

    #[test]
    fn test_has_metadata_with_valid_data() {
        let json = r#"{"model": {}}"#;
        let data = make_dvm_with_metadata(json, None);
        assert!(has_metadata(&data));
    }

    #[test]
    fn test_input_spec_dimensions() {
        let spec = InputSpec {
            size: Some("640x480".to_string()),
            input_channels: Some(3),
            output_channels: None,
            cameraadaptor: None,
        };
        assert_eq!(spec.dimensions(), Some((640, 480)));

        let no_size = InputSpec {
            size: None,
            input_channels: None,
            output_channels: None,
            cameraadaptor: None,
        };
        assert_eq!(no_size.dimensions(), None);

        let bad_size = InputSpec {
            size: Some("invalid".to_string()),
            input_channels: None,
            output_channels: None,
            cameraadaptor: None,
        };
        assert_eq!(bad_size.dimensions(), None);
    }
}
