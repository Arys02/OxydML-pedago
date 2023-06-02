use std::fmt::{Display, Formatter, write};
use ndarray::{array, Array, Array1, Array2, ArrayView, Axis, concatenate, Ix2, s, ShapeBuilder, stack};
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

    //TODO
    activation : fn(a: &f64) -> f64,
    seed: i32,
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
            W : weight,
            seed : 10,
            activation : sig
        }
    }

    pub fn _forward_propagate(&self, X_example : Array1<f64>) -> Array1<f64> {

        let mut a = X_example.clone();
        for i in 0..(self.layers.len() - 1){
            let bias : Array1<f64> = array![1.0];
            a = concatenate![Axis(0), bias, a];
            let mut z = self.W[i].t().dot(&a);
            println!("a : {:?} \nw : {:?}\nz: {:?}", a, self.W[i],z);
            a = z.map(sig);
        }
        a
    }
/*
    fn _back_forward_propagate(&self, x: Array1<f64>, y: Array1<f64>, alpha: f64) {
        let mut a = Vec::with_capacity(self.layers.l());
        let mut z = Vec::with_capacity(self.layers.l());
        let mut Δ =  Vec::with_capacity(self.layers.l());

        //First layers is input
        a.push(x.clone());

        let bias : Array1<f64> = array![1.0];

        //Forward propagation to fill z and a
        for i in 0..(self.layers.len() - 1){
            let next_a_with_bias = concatenate![Axis(0), bias, a[i].clone()];
            let current_z = self.W[i].t().dot(&next_a_with_bias);
            println!("a : {:?} \nw : {:?}\nz: {:?}", next_a_with_bias, self.W[i],z);

            a.push(current_z.map(sig));
            z.push(current_z);
            Δ.push(Array::zeros((1, current_z.len()).f()))
        }

        //Compute y predicted with propagation
        let y_pred: &Array1<f64> = &a[a.len() - 1];

        //Compute error
        let mut error = y_pred - &y;


        //compute delta on backpropagation for each layers
        for i in (0..self.layers.len() - 1).rev() {
            let prev_a_with_bias = concatenate![Axis(0), bias, a[i].clone()];
            Δ[i] += error.dot(a[i].t());
            error = prev_a_with_bias - y_pred;
        }
    }
 */
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
