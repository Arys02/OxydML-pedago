
pub trait Model {
    fn get_trained_variable(&self) -> **mut i32;

    fn predict(&self) -> **mut i32;

    fn fit(&self, X_train: **mut i32, Y_train: **mut i32, epoch: i32, alpha: f32){

    }
}

struct LinearRegressionModel {
    W : Vec<i32>
}

impl Model for LinearRegressionModel {
    fn get_trained_variable(&self) {
        todo!()
    }

    fn predict(&self) -> **mut i32 {
        todo!()
    }
}

#[no_mangle]
extern "C" fn create_linear_model(input_array: **mut i32,
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