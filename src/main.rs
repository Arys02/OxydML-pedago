mod plot_utils;
mod math_utils;
mod MLModel;
mod LinearRegression;
mod TestUtils;

fn main() {
    let mut x = 32;
    x = x + 1;
    plot_utils::plot_data(String::from("dataset/ex2data1.txt"));
    println!("Hello, world!") ;
}
