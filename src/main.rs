mod plot_utils;
mod math_utils;
mod MLModel;
mod LinearRegression;
mod TestUtils;

fn main() {
    plot_utils::plot_data(String::from("dataset/simple_linear.txt"),
                          [0.0, 40.0],
                          [0.0, 40.0], false);
    println!("Hello, world!");
}
