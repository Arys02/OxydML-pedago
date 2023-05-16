use ndarray::{array, Array1, Array2, ArrayView, Ix2, s};
use crate::MLModel::Model;
use crate::TestUtils;
use crate::math_utils::sig;

pub struct LinearRegressionModel {
    pub(crate) W : Array1<f64>
}

impl Model for LinearRegressionModel {
    fn _get_trained_variable(&self) -> &Array1<f64> {
        &self.W
    }

    // Gradiant descent for multiple features
    fn _fit(&mut self, X_train: Array2<f64>, Y_train: Array2<f64>, epoch: i32, alpha: f64, is_classification: bool) {
        let m  = X_train.nrows(); // number of training example
        let mf : f64 = m as f64;        // number of training example as float 64

        for _ in 0..epoch {
            let mut Wbis = self.W.clone();
            for j in 0..(self.W.shape()[0]) {
                let mut w = Wbis[j];
                let mut sum = 0.;

                for i in 0..m {
                    let X_i = &X_train.slice(s![i, ..]);
                    let Y_i = &Y_train.slice(s![i, ..]);
                    let mut hyp = self.W.dot(X_i);

                    if is_classification {
                        hyp = sig(&hyp);
                    }

                    let loss = hyp - Y_i;

                    sum += loss[[0]] * X_train[[i, j]];
                    println!("sum : {}", sum);
                }
                Wbis[j] = w - (alpha / mf) * sum;
                println!("Weight UPDATED : {}", Wbis[j]);
            }
            self.W = Wbis
        }
    }

    fn predict(&self) -> Array1<f64> {
        return array![1., 2.]
    }



    /*
    fn predict(&self) -> *mut f64 {
        TestUtils::fakeoutput_f64()
    }

     */
}
