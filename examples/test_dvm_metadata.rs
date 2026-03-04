use ara2::dvm_metadata::{has_metadata, read_labels_from_file, read_metadata_from_file};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dvm_path = if args.len() > 1 {
        Path::new(&args[1])
    } else {
        Path::new("/tmp/model.dvm")
    };

    println!("Testing DVM: {}", dvm_path.display());

    let data = std::fs::read(dvm_path).expect("Failed to read DVM file");
    println!("File size: {} bytes", data.len());
    println!("Has metadata: {}", has_metadata(&data));

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

    match read_labels_from_file(dvm_path) {
        Ok(labels) => {
            println!("\nLabels: {:?}", labels);
        }
        Err(e) => println!("Error reading labels: {}", e),
    }
}
