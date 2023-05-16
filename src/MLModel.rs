use ndarray::{array, Array1, Array2, ArrayView, Ix2};
use crate::LinearRegression::LinearRegressionModel;
use crate::{LinearRegression};

pub trait Model {
    fn _get_trained_variable(&self) -> &Array1<f64>;

    fn _fit(&mut self, X_train:  Array2<f64>, Y_train: Array2<f64>, epoch: i32, alpha: f64, is_classification: bool);

    fn predict(&self) -> Array1<f64>;
}

/*
#[no_mangle]
extern "C" fn create_linear_model(input_array: *mut i32,
                                  input_nb_elt: i32,
                                  input_elt_size: i32,
                                  is_classification: i32
) -> *mut LinearRegressionModel {

    let model = Box::new(LinearRegressionModel{
        W: array![],
    });

    let leaked = Box::leak(model);
    leaked
}

 */