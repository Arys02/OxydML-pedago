use ndarray::{Array1, Array2};

fn sig(a: &f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}

pub fn predict(ws: &Array1<f64>, xs: &Array2<f64>, is_classification: bool) -> Array1<f64> {
    xs.dot(ws).t().map(sig)
}

//pub fn gradiant_descente(teta : &Array1, pas: i32, m: usize)