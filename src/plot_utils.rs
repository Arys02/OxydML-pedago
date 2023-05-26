use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use ndarray_csv::Array2Reader;
use plotters::prelude::*;
use ndarray::{Array, array, Array2, ArrayBase, Axis, concatenate, Ix2, OwnedRepr};
use crate::LinearRegression::LinearRegressionModel;
use crate::MLModel::Model;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::build_dataset_utils::build_fake_dataset;


pub(crate) fn plot_data_array(x2: Array2<f64>, shape_len_x: [f64; 2], shape_len_y: [f64; 2], is_classif: bool){

    let root_area = BitMapBackend::new("out/etset.png", (600, 400))
        .into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Line Plot Demo", ("sans-serif", 40))
        .build_cartesian_2d(shape_len_x[0]..shape_len_x[1], shape_len_y[0]..shape_len_y[1])
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    if is_classif {
        ctx.draw_series(
            x2.outer_iter().filter(|val| val[[val.len()-1]] == 0.0).map(|val| TriangleMarker::new((val[[0]], val[[1]]), 5, &RED))
        ).unwrap();

        ctx.draw_series(

            x2.outer_iter().filter(|val| val[[val.len() - 1]] == 1.0).map(|val| Circle::new((val[[0]], val[[1]]), 5, &BLUE))
        ).unwrap();
    }

    else {
        ctx.draw_series(
            x2.outer_iter().map(|val| TriangleMarker::new((val[[0]], val[[1]]), 5, &RED))
        ).unwrap();
    }


}

pub(crate) fn plot_data(data_src: String, shape_len_x: [f64; 2], shape_len_y: [f64; 2], is_classif: bool){
    let x2: Array2<f64> = get_csv2_f64(&data_src).unwrap();
    let x2 = build_fake_dataset(2., 3., 5.);

    let (x1 , y1) = x2.view().split_at(Axis(1), x2.shape()[1] - 1);
    let x1_1: Array2<f64> = Array::<f64, Ix2>::from_elem((x1.shape()[0], 1), 1.);
    let x1_concatenated: Array2<f64> = concatenate![Axis(1), x1_1, x1];

    println!(" X1 : concatenate {:?}", x1_concatenated);
    let mut linear_r = LinearRegressionModel{ W: array![0.1, 0.1, 0.1] };
    linear_r._fit(x1_concatenated, y1.into_owned(), 10, 0.0001, is_classif);
    let result_w = linear_r._get_trained_variable();
    let result_points = Array::from_shape_fn((40, 2),
                                             |(i , j ) |
                                                 ((if j == 1 {0.} else {1.}) * ((i as f64) + 1.) + (j as f64) * (((i as f64) + 1.) * result_w[1] + result_w[0])));

    let result_points_draw = Array::from_shape_fn((40, 1),
                                             |(i, j)  |
                                                 ((i as f64) + 1., (((i as f64) + 1.) * result_w[1] + result_w[0])));

    println!("{}", linear_r._get_trained_variable());
    println!("result points : {}", result_points);

    let root_area = BitMapBackend::new("out/etset.png", (600, 400))
        .into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Line Plot Demo", ("sans-serif", 40))
        .build_cartesian_2d(shape_len_x[0]..shape_len_x[1], shape_len_y[0]..shape_len_y[1])
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    if is_classif {
        ctx.draw_series(
            x2.outer_iter().filter(|val| val[[val.len()-1]] == 0.0).map(|val| TriangleMarker::new((val[[0]], val[[1]]), 5, &RED))
        ).unwrap();
        ctx.draw_series(

            x2.outer_iter().filter(|val| val[[val.len() - 1]] == 1.0).map(|val| Circle::new((val[[0]], val[[1]]), 5, &BLUE))
        ).unwrap();
    }


    else {
         ctx.draw_series(
            x2.outer_iter().map(|val| TriangleMarker::new((val[[0]], val[[1]]), 5, &RED))
        ).unwrap();
    }
    //draw result
    ctx.draw_series(LineSeries::new(
        result_points_draw,

        &GREEN,
    )).unwrap();


}

pub fn get_csv2_f64(src : &String) -> Result<ArrayBase<OwnedRepr<f64>, Ix2>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(src)?;

    let mut data = Vec::new();
    let mut data2 = Vec::new();

    for result in rdr.records() {
        let record = result?;
        data.push(record.iter().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<f64>>());
    }

    let ncol = data.first().map_or(0, |row| row.len());
    let mut nrows = 0;

    for i in 0..data.len() {
        data2.extend_from_slice(&data[i]);
        nrows += 1;
    }
    let array = Array2::from_shape_vec((nrows, ncol), data2).unwrap();
    Ok(array)
}

fn get_csv2_f32(src : &String) -> Result<Array2<f32>, Box<dyn Error>>{
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(src)?;

    let mut data = Vec::new();
    let mut data2 = Vec::new();

    for result in rdr.records() {
        let record = result?;
        data.push(record.iter().map(|s| s.parse::<f32>().unwrap()).collect::<Vec<f32>>());
    }

    let ncol = data.first().map_or(0, |row| row.len());
    let mut nrows = 0;

    for i in 0..data.len(){
        data2.extend_from_slice(&data[i]);
        nrows += 1;
    }
    let array = Array2::from_shape_vec((nrows, ncol), data2).unwrap();
    println!("{}",array);
    Ok(array)
}

fn get_csv(src : &String) -> Result<Vec<Vec<f64>>, Box<dyn Error>>{
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(src)?;

    let mut data = Vec::new();
    //let x = rdr.deserialize_array2((97, 2));

    //let y = rdr.headers().unwrap().len();



    //let mut array_read = rdr.deserialize_array2((x, y));


    for result in rdr.records() {
        let record = result?;
        data.push(record.iter().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<f64>>());
    }
    Ok(data)
}