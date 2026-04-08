//! # DVM Metadata Inspector
//!
//! Reads and displays the EdgeFirst metadata embedded in a compiled DVM
//! (Deep Vision Model) file.  DVM files are ZIP archives containing the
//! compiled neural network plus an `edgefirst.json` metadata sidecar and
//! an optional `labels.txt` class label list.
//!
//! This example does not require ARA-2 hardware — it only parses the
//! metadata from the file on disk.
//!
//! ## Usage
//!
//! ```text
//! test_dvm_metadata <model.dvm>
//! ```

use ara2::dvm_metadata::{has_metadata, read_labels_from_file, read_metadata_from_file};
use std::path::Path;

/// Read a DVM file and print all embedded EdgeFirst metadata.
///
/// Extracts and displays:
/// - **Task** — detect, segment, classify, or pose
/// - **Classes** — the label list embedded in the model
/// - **Input spec** — resolution, channel count, camera adaptor format
/// - **Model info** — size variant (n/s/m/l/x), detection/segmentation flags
/// - **Deployment info** — model name and author
/// - **Compilation info** — target device, PPA metrics (IPS, power)
/// - **Output specs** — per-output name, type, and shape
/// - **Labels** — the full class label list from `labels.txt`
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dvm_path = if args.len() > 1 {
        Path::new(&args[1])
    } else {
        Path::new("/tmp/model.dvm")
    };

    println!("Testing DVM: {}", dvm_path.display());

    // Check the raw ZIP for the metadata marker before full parsing.
    let data = std::fs::read(dvm_path).expect("Failed to read DVM file");
    println!("File size: {} bytes", data.len());
    println!("Has metadata: {}", has_metadata(&data));

    // Parse the full metadata JSON from the DVM archive.
    match read_metadata_from_file(dvm_path) {
        Ok(Some(meta)) => {
            println!("\nMetadata found:");
            println!("  task: {:?}", meta.task());
            println!("  classes: {:?}", meta.classes());

            if let Some(input) = &meta.input {
                if let Some((w, h)) = input.dimensions() {
                    println!("  input: {}x{} ch={:?}", w, h, input.input_channels);
                }
                println!("  camera adaptor: {:?}", input.cameraadaptor);
            }

            if let Some(model) = &meta.model {
                println!("  model_size: {:?}", model.model_size);
                println!("  segmentation: {}", model.segmentation);
                println!("  detection: {}", model.detection);
            }

            if let Some(deployment) = &meta.deployment {
                println!("  model_name: {:?}", deployment.model_name);
            }

            if let Some(compilation) = &meta.compilation {
                println!("  compilation.target: {:?}", compilation.target);
                if let Some(ppa) = &compilation.ppa {
                    println!("  ppa.ips: {:?}", ppa.ips);
                    println!("  ppa.power_mw: {:?}", ppa.power_mw);
                }
            }

            println!("\n  outputs:");
            for out in &meta.outputs {
                println!(
                    "    - name: {:?}, type: {:?}, shape: {:?}",
                    out.name, out.output_type, out.shape
                );
            }
        }
        Ok(None) => println!("No metadata found"),
        Err(e) => println!("Error: {}", e),
    }

    // Read the class labels from the embedded labels.txt file.
    match read_labels_from_file(dvm_path) {
        Ok(labels) => {
            println!("\nLabels: {:?}", labels);
        }
        Err(e) => println!("Error reading labels: {}", e),
    }
}
