use std::error::Error;
use std::fs::File;
use csv::Reader;
use fast_float::parse;
use plotters::prelude::*;
use ndarray::{Array, array, Array2};
use ndarray_csv::{Array2Reader, Array2Writer};


pub(crate) fn plot_data(data_src: String){
    let x = get_csv(data_src);

    let vec = match x {
        Ok(vec) => vec,
        _ => Vec::new()
    };


    let root_area = BitMapBackend::new("out/etset.png", (600, 400))
        .into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Line Plot Demo", ("sans-serif", 40))
        .build_cartesian_2d(20.0..100.0, 20.0..100.0)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(
       vec.iter().filter(|val| val[2] == 0.0).map(|val| TriangleMarker::new((val[0], val[1]), 5, &RED))
    ).unwrap();

    ctx.draw_series(
       vec.iter().filter(|val| val[2] == 1.0).map(|val| Circle::new((val[0], val[1]), 5, &BLUE))
    ).unwrap();

}

fn get_csv(src : String) -> Result<Vec<Vec<f64>>, Box<dyn Error>>{
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(src)?;

    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result?;
        data.push(record.iter().map(|s| s.parse::<f64>().unwrap()).collect::<Vec<f64>>());
    }
    Ok(data)
}