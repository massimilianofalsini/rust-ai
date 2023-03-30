extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use csv::ReaderBuilder;
use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::Zip;
use ndarray_csv::Array2Reader;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;
use std::fs::File;

fn get_csv_shape() -> Vec<usize> {
    // TO DO
    vec![42000, 785]
}

fn init_params(csv_shape: Vec<usize>) -> Vec<Array2<f32>> {
    let w1 = Array::random((10, csv_shape[1] - 1), Uniform::new(-0.5, 0.5));
    let b1 = Array::random((10, 1), Uniform::new(-0.5, 0.5));
    let w2 = Array::random((10, 10), Uniform::new(-0.5, 0.5));
    let b2 = Array::random((10, 1), Uniform::new(-0.5, 0.5));
    vec![w1, b1, w2, b2]
}

fn re_l_u(z: Array2<f32>) -> Array2<f32> {
    let mut x = Array2::<f32>::zeros((z.shape()[0], z.shape()[1]));
    Zip::from(&mut x).and(&z).for_each(|a, &b| {
        if b > 0.0 {
            *a = b
        }
    });
    x
}

fn softmax(z: Array2<f32>) -> Array2<f32> {
    let x = z.clone();
    let x_to_the_e = x.mapv(f32::exp);
    let sum_of_x_to_the_e: f32 = x_to_the_e.clone().into_iter().sum();
    let softmax = x_to_the_e.map(|k| k / sum_of_x_to_the_e);
    softmax
}

fn forward_propagation(w1: Array2<f32>, b1: Array2<f32>, w2: Array2<f32>, b2: Array2<f32>) {
    let x = Array2::<f32>::zeros((1, 1)); // TO DO

    let z1 = w1.dot(&x) + b1;
    let a1 = re_l_u(z1);
    let z2 = w2.dot(&a1) + b2;
    let a2 = softmax(z2);
}

fn one_hot(y: Array2<f32>) -> Array2<f32> {
    // TO DO
    y
}

fn re_l_u_derivative(z: Array2<f32>) -> Array2<f32> {
    let mut x = z.clone();
    for k in x.iter_mut() {
        if *k > 0.0 {
            *k = 1.0
        }
    }
    x
}

fn backward_prop(
    z1: Array2<f32>,
    a1: Array2<f32>,
    // z2: Array2<f32>,
    a2: Array2<f32>,
    w2: Array2<f32>,
    y: Array2<f32>,
    x: Array2<f32>,
) -> (Array2<f32>, f32, Array2<f32>, Array1<f32>) {
    let m = y.len() as f32;
    let one_hot_y = one_hot(y);
    let dz2 = a2 - one_hot_y;
    let dw2 = 1.0 / m * dz2.dot(&a1.t());
    let db2 = 1.0 / m + dz2.sum_axis(Axis(2));
    let dz1 = w2.t().dot(&dz2) * re_l_u_derivative(z1);
    let dw1 = 1.0 / m * dz1.dot(&x.t());
    let db1 = 1.0 / m + dz1.sum();
    (dw1, db1, dw2, db2)
}

fn update_param(
    w1: Array2<f32>,
    b1: Array2<f32>,
    w2: Array2<f32>,
    b2: Array2<f32>,
    dw1: Array2<f32>,
    db1: Array2<f32>,
    dw2: Array2<f32>,
    db2: Array2<f32>,
    alpha: f32,
) -> Vec<Array2<f32>> {
    let new_w1 = w1 - alpha * dw1;
    let new_b1 = b1 - alpha * db1;
    let new_w2 = w2 - alpha * dw2;
    let new_b2 = b2 - alpha * db2;
    vec![new_b1, new_b2, new_w1, new_w2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let csv_shape = get_csv_shape();

    // Read an array back from the file
    let file = File::open("mnist_train.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let array_read: Array2<u8> = reader.deserialize_array2((csv_shape[0], csv_shape[1]))?;

    // Init params
    let params = init_params(csv_shape);

    Ok(())
}
