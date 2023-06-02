use ndarray::{array, Axis, s};
use crate::LinearRegression::LinearRegressionModel;
use crate::build_dataset_utils::{build_fake_dataset, build_fake_dataset_no_bias, write_dataset};
use crate::MLModel::Model;
use crate::multi_layer_perceptron::MultiLayerPerceptron;
use crate::plot_utils::get_csv2_f64;

mod plot_utils;
mod build_dataset_utils;
mod MLModel;
mod LinearRegression;
mod TestUtils;
mod multi_layer_perceptron;


fn main() {
    /*
    plot_utils::plot_data(String::from("dataset/ex1data2.txt"),
                          [0.0, 40.0],
                          [0.0, 40.0], false);
     */

    //let x2 = build_fake_dataset_no_bias(2., 3., 5.);
    //write_dataset("dataset/test_no_bias.txt", x2.clone()).expect("TODO: panic message");

    let X = get_csv2_f64(&String::from("dataset/simple_linear.txt")).unwrap();

    let (x1 , y1) = X.view().split_at(Axis(1), X.shape()[1] - 1);

    println!("X == {:?}", X);
    println!("{}", y1);

    let mut linear_r = LinearRegressionModel{ W: array![0.1, 0.1] };
    linear_r._fit(x1.into_owned(), y1.into_owned(), 5000, 0.01, false);

    let result_w = linear_r.predict(array![5.0]);
    println!("{} Hello, world!", result_w);


    //let mut mlp : MultiLayerPerceptron = MultiLayerPerceptron::new(vec![2, 2, 1]);
    //mlp._forward_propagate(x1.slice(s![0, ..]).into_owned());


    //println!("{}", mlp);


}
