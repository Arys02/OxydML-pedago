use std::ptr::null_mut;
use ndarray::{Array, array, Array1, Array2};
use crate::LinearRegression::LinearRegressionModel;
use crate::TestUtils;

pub trait Model {
    fn _get_trained_variable(&self) -> Array1<i32>;

    fn _fit(&self, X_train: Array2<i32>, Y_train: Array2<i32>, epoch: i32, alpha: f32){

    }

    fn predict(&self) -> *mut f32;
}

#[no_mangle]
extern "C" fn create_linear_model(input_array: *mut i32,
                                  input_nb_elt: i32,
                                  input_elt_size: i32,
                                  is_classification: i32
) -> *mut LinearRegressionModel {

    let model = Box::new(LinearRegressionModel{
        W: vec![],
    });

    let leaked = Box::leak(model);
    leaked
}