use std::ptr::null_mut;
use ndarray::{Array, array, Array1, Array2};

pub trait Model {
    fn _get_trained_variable(&self) -> Array1<i32>;

    fn _fit(&self, X_train: Array2<i32>, Y_train: Array2<i32>, epoch: i32, alpha: f32){

    }

    fn predict(&self) -> *mut f32;
}

fn fakeoutput_i32() -> *mut i32 {
    let fake = vec![10];
    fake.leak().as_mut_ptr()
}
fn fakeoutput_f32() -> *mut f32 {
    let fake = vec![10.0];
    fake.leak().as_mut_ptr()
}


struct LinearRegressionModel {
    W : Vec<i32>
}

impl Model for LinearRegressionModel {
    fn _get_trained_variable(&self) -> Array1<i32> {
        return array![0]
    }

    fn _fit(&self, X_train: Array2<i32>, Y_train: Array2<i32>, epoch: i32, alpha: f32) {
        todo!()
    }

    fn predict(&self) -> *mut f32 {
        fakeoutput_f32()
    }
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