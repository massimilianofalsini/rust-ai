extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use csv::ReaderBuilder;
use ndarray::Array;
use ndarray::Array2;
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

fn softmax(z: Array2<f32>) {
    // TO DO
}

fn forward_propagation(w1: Array2<f32>, b1: Array2<f32>, w2: Array2<f32>, b2: Array2<f32>) {
    let x = Array2::<f32>::zeros((1, 1)); // TO DO

    let z1 = w1.dot(&x) + b1;
    let a1 = re_l_u(z1);
    let z2 = w2.dot(&a1) + b2;
    let a2 = softmax(z2);
}

fn main() -> Result<(), Box<dyn Error>> {
    let csv_shape = get_csv_shape();

    // Read an array back from the file
    let file = File::open("mnist_train.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let array_read: Array2<u8> = reader.deserialize_array2((csv_shape[0], csv_shape[1]))?;
    println!("{}", array_read);

    // Init params
    let params = init_params(csv_shape);
    let w1 = params[0].clone();
    let b1 = params[1].clone();
    let w2 = params[2].clone();
    let b2 = params[3].clone();
    println!("{}", params[2].clone());

    // Forward propagation
    forward_propagation(w1, b1, w2, b2);

    Ok(())
}
