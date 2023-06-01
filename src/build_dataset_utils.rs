use std::error::Error;
use std::fs::File;
use csv::{ReaderBuilder, WriterBuilder};
use ndarray::{Array1, Array2};
use ndarray_csv::{Array2Reader, Array2Writer};
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub(crate) fn sig(a: &f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}

pub (crate) fn build_fake_dataset_polynomial(a : f64, b : f64, c: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut normal = Normal::new(0.0, 1.0).unwrap();
    let SIZE = 50;

    // Initialize an empty 2D array with dimensions (SIZE, 3).
    let mut data = Array2::<f64>::zeros((SIZE, 3));

    for i in 0..SIZE {
        // Generate a random x value.
        let x = rng.gen::<f64>();

        // Compute y = ax^2 + bx + c + noise.
        let y = a * x.powi(2) + b * x + c + normal.sample(&mut rng);

        // Assign x and y to the ith row of the data array.
        data[[i, 0]] = 1.;  // the bias term
        data[[i, 1]] = x;
        data[[i, 2]] = y;
    }

    data

}

pub (crate) fn build_fake_dataset_no_bias(a : f64, b : f64, c: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let SIZE = 50;
    let mut normal = Normal::new(0.0, 0.3).unwrap();

    // Initialize an empty 2D array with dimensions (SIZE, 3).
    let mut data = Array2::<f64>::zeros((SIZE, 3));

    for i in 0..SIZE {

        // Generate a random x value.
        let x1 = rng.gen::<f64>();
        let x2 = rng.gen::<f64>();

        // Compute y = ax^2 + bx + c + noise.
        let y = a * x1 + b * x2 + c + normal.sample(&mut rng);

        // Assign x and y to the ith row of the data array.
        data[[i, 0]] = x1;
        data[[i, 1]] = x2;
        data[[i, 2]] = y;
    }
    data
}

pub (crate) fn build_fake_dataset(a : f64, b : f64, c: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let SIZE = 50;
    let mut normal = Normal::new(0.0, 0.3).unwrap();

    // Initialize an empty 2D array with dimensions (SIZE, 3).
    let mut data = Array2::<f64>::zeros((SIZE, 4));

    for i in 0..SIZE {

        // Generate a random x value.
        let x1 = rng.gen::<f64>();
        let x2 = rng.gen::<f64>();

        // Compute y = ax^2 + bx + c + noise.
        let y = a * x1 + b * x2 + c + normal.sample(&mut rng);

        // Assign x and y to the ith row of the data array.
        data[[i, 0]] = 1.;  // the bias term
        data[[i, 1]] = x1;
        data[[i, 2]] = x2;
        data[[i, 3]] = y;
    }

    data

}

pub (crate) fn write_dataset(filename : &str, arrayds: Array2<f64>) ->Result <(), Box<dyn Error>> {
    {
        let file = File::create(filename)?;
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        writer.serialize_array2(&arrayds)?;
    }
    Ok(())
}

