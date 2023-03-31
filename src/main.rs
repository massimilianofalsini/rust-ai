extern crate csv;
extern crate ndarray;

use csv::ReaderBuilder;
use ndarray::s;
use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Zip;
use ndarray_csv::Array2Reader;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;
use std::fs::File;

fn get_csv_shape() -> (usize, usize) {
    // TO DO
    (42000, 785)
}

fn init_params(columns: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let w1 = Array::random((10, columns - 1), Uniform::new(-0.5, 0.5));
    let b1 = Array::random((10, 1), Uniform::new(-0.5, 0.5));
    let w2 = Array::random((10, 10), Uniform::new(-0.5, 0.5));
    let b2 = Array::random((10, 1), Uniform::new(-0.5, 0.5));
    (w1, b1, w2, b2)
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

fn forward_prop(
    w1: Array2<f32>,
    b1: Array2<f32>,
    w2: Array2<f32>,
    b2: Array2<f32>,
    x: Array2<f32>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let z1 = w1.dot(&x) + b1;
    let a1 = re_l_u(z1.clone());
    let z2 = w2.dot(&a1) + b2;
    let a2 = softmax(z2.clone());
    (z1, a1, z2, a2)
}

fn one_hot(y: Array1<f32>) -> Array2<f32> {
    let mut max = 0.0;
    for elem in y.iter() {
        if elem > &max {
            max = *elem;
        }
    }
    let mut one_hot_y = Array2::<f32>::zeros((y.len(), max as usize + 1));
    for (i, elem) in y.iter().enumerate() {
        // inverting indices to not transpose after
        one_hot_y[[*elem as usize, i]] = 1.0;
    }
    one_hot_y
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
    _z2: Array2<f32>,
    a2: Array2<f32>,
    _w1: Array2<f32>,
    w2: Array2<f32>,
    x: Array2<f32>,
    y: Array1<f32>,
) -> (Array2<f32>, f32, Array2<f32>, f32) {
    let m = y.len() as f32;
    let one_hot_y = one_hot(y);
    let dz2 = a2 - one_hot_y;
    let dw2 = 1.0 / m * dz2.dot(&a1.t());
    let db2 = 1.0 / m + dz2.sum();
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
    db1: f32,
    dw2: Array2<f32>,
    db2: f32,
    alpha: f32,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let new_w1 = w1 - alpha * dw1;
    let new_b1 = b1 - alpha * db1;
    let new_w2 = w2 - alpha * dw2;
    let new_b2 = b2 - alpha * db2;
    (new_b1, new_b2, new_w1, new_w2)
}

fn get_predictions(a2: Array2<f32>) -> usize {
    let (mut index, mut max) = (0, 0.0);
    for (i, elem) in a2.iter().enumerate() {
        if (index, &max) < (0, elem) {
            max = *elem;
            index = i;
        }
    }
    index
}

fn get_accuracy(predictions: usize, y: Array1<f32>) -> usize {
    let mut sum = 0;
    for elem in y.iter() {
        if predictions as f32 == *elem {
            sum += 1;
        }
    }
    let accuracy = sum / y.len();
    accuracy
}

fn gradient_descent(x: Array2<f32>, y: Array1<f32>, alpha: f32, iterations: usize, columns: usize) {
    let (w1, b1, w2, b2) = init_params(columns);
    for i in 0..iterations {
        // error: infinite loop
        let (z1, a1, z2, a2) =
            forward_prop(w1.clone(), b1.clone(), w2.clone(), b2.clone(), x.clone());
        let (dw1, db1, dw2, db2) = backward_prop(
            z1,
            a1,
            z2,
            a2.clone(),
            w1.clone(),
            w2.clone(),
            x.clone(),
            y.clone(),
        );
        let (_b1, _b2, _w1, _w2) = update_param(
            w1.clone(),
            b1.clone(),
            w2.clone(),
            b2.clone(),
            dw1,
            db1,
            dw2,
            db2,
            alpha,
        );
        if i % 10 == 0 {
            println!("Iteration {}", i);
            let predictions = get_predictions(a2.clone());
            println!("{}", get_accuracy(predictions, y.clone()));
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let (rows, columns) = get_csv_shape();

    // read a matrix back from the file
    let file = File::open("mnist_train.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let array_read: Array2<u8> = reader.deserialize_array2((rows, columns))?;
    println!("csv readed successfully");
    // after transposition rows are columns and vice versa
    let data_train = array_read.t();
    // get the first row
    let y_train = data_train.slice(s![0, ..]).mapv(|elem| elem as f32);
    //get the other of the rows and normalize them
    let x_train = data_train
        .slice(s![1..columns, ..])
        .mapv(|elem| elem as f32 / 255.0);
    gradient_descent(x_train, y_train, 0.1, 500, columns);
    Ok(())
}
