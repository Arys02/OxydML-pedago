use ndarray::{array, Array1, Array2};
use crate::MLModel::Model;
use crate::TestUtils;

pub struct LinearRegressionModel {
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
        TestUtils::fakeoutput_f32()
    }
}
