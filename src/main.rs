extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use csv::ReaderBuilder;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;

fn get_csv_shape() -> (usize, usize) {
    // TO DO
    (10000, 785)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Read an array back from the file
    let file = File::open("mnist_train.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);

    let csv_shape = get_csv_shape();
    let array_read: Array2<u64> = reader.deserialize_array2(csv_shape)?;

    println!("{}", array_read);
    Ok(())
}
