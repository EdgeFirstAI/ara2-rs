use ara2::Session;
use criterion::{Criterion, criterion_group, criterion_main};
use edgefirst_hal::{
    image::{Crop, Flip, G2DProcessor, ImageProcessorTrait as _, Rotation, load_image},
    tensor::{PixelFormat, Tensor, TensorDyn, TensorMemory, TensorTrait as _},
};
use std::{env, path::Path};

fn model_benchmark(c: &mut Criterion) {
    let session = Session::create_via_unix_socket("/var/run/ara2.sock").unwrap();
    let endpoint = session.list_endpoints().unwrap().pop().unwrap();
    let modelpath = env::var("MODEL")
        .unwrap_or_else(|_| "testdata/yolov8s_seg_960x544_rgba_nhwc.dvm".to_string());
    let modelpath = Path::new(&modelpath);
    let mut model = endpoint.load_model_from_file(modelpath).unwrap();
    model.allocate_tensors(None).unwrap();

    let image = env::var("IMAGE").unwrap_or_else(|_| "testdata/zidane.jpg".to_string());

    c.bench_function("load_image", |b| {
        b.iter(|| {
            let file = std::fs::read(&image).expect("Failed to read image file");
            load_image(&file, Some(PixelFormat::Rgba), Some(TensorMemory::Dma))
                .expect("Failed to load image");
        });
    });

    let file = std::fs::read(&image).expect("Failed to read image file");
    let tensor = load_image(&file, Some(PixelFormat::Rgba), Some(TensorMemory::Dma))
        .expect("Failed to load image");
    let mut converter = G2DProcessor::new().expect("Failed to create G2DProcessor");

    c.bench_function("convert_image", |b| {
        b.iter(|| {
            let input_tensor = Tensor::<u8>::from_fd(
                model.input_tensor(0).clone_fd().unwrap(),
                model.input_tensor(0).shape(),
                None,
            )
            .expect("Failed to create input tensor from file descriptor");
            let mut dst = TensorDyn::from(input_tensor)
                .with_format(PixelFormat::Rgba)
                .expect("Failed to create destination tensor image");
            converter
                .convert(
                    &tensor,
                    &mut dst,
                    Rotation::None,
                    Flip::None,
                    Crop::default(),
                )
                .expect("Failed to convert image");
        });
    });

    c.bench_function(modelpath.file_stem().unwrap().to_str().unwrap(), |b| {
        b.iter(|| {
            model.run().unwrap();
        });
    });

    println!("Model Stats: {:?}", model.run().unwrap());
}

criterion_group!(model, model_benchmark);
criterion_main!(model);
