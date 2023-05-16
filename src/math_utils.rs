use ndarray::{Array1, Array2};

pub(crate) fn sig(a: &f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}