use std::fmt::{Display, Formatter, write};
use ndarray::{array, Array, Array1, Array2, ArrayView, Ix2, s};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use crate::MLModel::Model;
use crate::TestUtils;
use crate::build_dataset_utils::sig;

static INIT_W : f64 = 0.01;


struct DenseLayer {

}


pub struct MultiLayerPerceptron {
    //vector with size of each layers
    pub layers : Vec<usize>,
    //pub

    //vector with every weight layers
    pub(crate) W : Vec<Array2<f64>>,
}


/*
impl Model for MultiLayerPerceptron {
    fn _get_trained_variable(&self) -> &Array1<f64> {
        let a = self.W[[0, 1]];
    }

    fn _fit(&mut self, X_train:  Array2<f64>, Y_train: Array2<f64>, epoch: i32, alpha: f64, is_classification: bool) {
        for _ in 0 .. epoch {
            //_forward_propagate()
        }
    }

    fn predict(&self) -> Array1<f64> {
        array![1.0]
    }
}
 */


impl MultiLayerPerceptron {
    pub fn new(layers: Vec<usize>) -> MultiLayerPerceptron {
        let mut weight = Vec::new();

        for i in 0..(layers.len() - 1) {
            let layer_weight = Array2::random(
                (layers[i] + 1 /* +1 for bias */, layers[i+1]),
                Uniform::new(-1.0, 1.0)
            );

            weight.push(layer_weight);
        }

        MultiLayerPerceptron {
            layers,
            W : weight
        }
    }

    fn _forward_propagate(&self, X_train : Array2<f64>, Y_train : Array2<f64>) {
        for i in 0..(self.layers.len() - 1){

        }
    }

    fn _back_forward_propagate() {}

}



impl Display for MultiLayerPerceptron {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "layers : \n \
        {:?} \n \
        weight : \n \
        ", self.layers).expect("TODO: panic message");

        for i in 0..(self.W.len()) {
            write!(f, "\n layer weight : {} \n {:?}", i, self.W[i]).expect("TODO: panic message");
        }
        write!(f, "")
    }
}
