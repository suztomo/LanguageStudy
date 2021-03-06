// Implementation of Neural Network for MNIST
// Python implementation: https://www.python-course.eu/neural_network_mnist.php
extern crate csv;
use std::fs::File;
use std::{thread, time};
use std::time::Instant;

extern crate ansi_term;
use ansi_term::Colour::*;
use ansi_term::{ANSIString, ANSIStrings, Style};

#[macro_use]
extern crate lazy_static; // 1.0.1;

#[macro_use]
extern crate ndarray;
//extern crate ndarray_linalg;
extern crate rand;
extern crate ndarray_rand;

use std::string::ToString;
use rand::{thread_rng, Rng};
use rand::distributions::Normal;
use ndarray_rand::{RandomExt, F32};
// use rand::Rng;
use ndarray::prelude::*;
extern crate utils;
use utils::math::sigmoid;


fn tanh(x: f32) -> f32 {
    x.tanh()
}

fn tanh_prime(x: f32) -> f32 {
    let t = tanh(x);
    1. - t*t
}

fn sigmoid_derivative(x: f32) -> f32 {
    // https://beckernick.github.io/sigmoid-derivative-neural-network/
    let t = sigmoid(x);
    t * (1.0 - t)
}


const ACTIVATION:&Fn(f32) -> f32 = &sigmoid;
const ACTIVATION_PRIME: &Fn(f32) -> f32 = &sigmoid_derivative;
// Tanh doesn't work as the range is -1 to + 1
//const ACTIVATION:&Fn(f32) -> f32 = &tanh;
//const ACTIVATION_PRIME: &Fn(f32) -> f32 = &tanh_prime;



const IMG_H_SIZE: usize = 28;
const IMG_W_SIZE: usize = 28;
//const HIDDEN_LAYER_SIZE: usize = 100;

const MNIST_DOT_MAX:f32 = 255.;

#[derive(Debug)]
struct NeuralNetwork<'a> {
    hidden_layer_size: usize,
    // <row count> x <column count>
    last_input: &'a Array2<f32>, // Input 1 x 784 (28x28)
    weight1: Array2<f32>,    // 784 x 100
    layer1_a: Array2<f32>,   // 1 x 100, before activation
    layer1: Array2<f32>,     // 1 x 100
    weight2: Array2<f32>,    // 100 x 10
    last_output_a: Array2<f32>, // 1 x 10. before activation. PRML uses 'a' for before activation. 'z' for after activation.
    last_output: Array2<f32>, // 1 x 10 // one-hot representation for 10 digits (0-9)
}

lazy_static! {
    static ref INPUT_ZERO : Array2<f32> = Array::zeros((1, IMG_H_SIZE*IMG_W_SIZE));
}


impl<'a> NeuralNetwork<'a> {

    pub fn new(hidden_layer_size: usize) -> NeuralNetwork<'a> {
        let nn = NeuralNetwork {
            last_input: &INPUT_ZERO,
            //weight1 : Array::zeros((IMG_H_SIZE*IMG_W_SIZE, HIDDEN_LAYER_SIZE)),
            weight1: Array::random((IMG_H_SIZE*IMG_W_SIZE, hidden_layer_size), F32(Normal::new(0., 1.))),
            //weight2 : Array::zeros((HIDDEN_LAYER_SIZE, 10)),
            weight2: Array::random((hidden_layer_size, 10), F32(Normal::new(0., 1.))),
            last_output_a: Array::zeros((1, 10)),
            last_output: Array::zeros((1, 10)),
            layer1_a: Array::zeros((1, hidden_layer_size)),
            layer1 : Array::zeros((1, hidden_layer_size)),
            hidden_layer_size,
        };
        nn
    }
    fn feed_forward(&mut self, input_array: &'a Array2<f32>) -> i32 {
        self.last_input = input_array; // input_array.to_owned().insert_axis(Axis(0));
        // When theres' wrong number of rows and column, you'll get following runtime error:

        // lastInput: 1 x 784, weight1: 784 x 100, and thus dot_product: 1 x 100
        self.layer1_a = self.last_input.dot(&self.weight1);
        // https://docs.rs/ndarray/0.11.2/ndarray/struct.ArrayBase.html#method.mapv
        // layer1: 1 x 100
        self.layer1 = self.layer1_a.mapv(ACTIVATION);
 
        // When the matrix size is different:
        //   'assertion failed: `(left == right)`
        //     left: `4`,
        //     right: `3`'
        // layer1: 1 x 100, weight2: 100 x 10, and thus output is 1 x 10 (Array2)
        // before activation
        self.last_output_a = self.layer1.dot(&self.weight2);
        // 1 x 10
        self.last_output = self.last_output_a.mapv(ACTIVATION);
        let label = LabelTable::array_to_label(self.last_output.view());
        label
    }

    // y: 1 x 10 matrix
    fn back_prop(&mut self, y: &Array2<f32>) {
        // Input layer     hidden layer       output layer
        //  784 -> (weights1) -> 100 -> (weights2) -> 10

        // PRML p.244 "Error Backpropagation"
        // 1. "Apply an input vector..." is already done at feedforward function

        // 2. Evaluate δ (delta) for all the output unit. In this case it's only one value
        // Get Error δ (delta) at the output, the output layer only has one element so f32 is enough.
        // delta_output: 1 x 10
        let delta_output:Array2<f32> = y - &self.last_output;

        // 3. Backpropagate deltas using next delta and weight
        // In this case, hidden Layer has 4 nodes. So the error (delta) is 4-element matrix
        //     self.layer1_a : 1 x 4 matrix
        // hidden_delta_0 = h'(self.layer1_a[[0, 0]]) * (weights2[[0, 0]] * delta_output[[0, 0]]
        //                                              +weights2[[0, 1]] * delta_output[[0, 1]]
        //                                              +...(10 elements, from 0 to 9)
        //                                              +weights2[[0, 9]] * delta_output[[0, 9]])
        // hidden_delta_1 = h'(self.layer1_a[[0, 1]]) * (weights2[[1, 0]] * delta_output[[0, 0]]
        //                                              +weights2[[1, 1]] * delta_output[[0, 1]]
        //                                              +...(10 elements, from 0 to 9)
        //                                              +weights2[[1, 9]] * delta_output[[0, 9]])
        // ...100 elements, from 0 to 99
        // hidden_delta_99 = h'(self.layer1_a[[0,99]])* (weights2[[99, 0]] * delta_output[[0, 0]]
        //                                              +weights2[[99, 1]] * delta_output[[0, 1]]
        //                                              +...(10 elements, from 0 to 9)
        //                                              +weights2[[99, 9]] * delta_output[[0, 9]])

        // self.weight2 is 100 x 10 matrix, delta_output is 1 x 10. delta_output.t() is 10 x 1
        // So weights2_by_delta_output is 100 x 1
        // For scalar-output, it was like the below with mapv
        //   let weights2_by_delta_output: Array2<f32> = self.weight2.mapv(|i| i * delta_output);
        let weights2_by_delta_output: Array2<f32> = self.weight2.dot(&delta_output.t());
        // self.layer1_a.t() is 100 x 1 matrix. So is layer1_a_derived.
        let layer1_t = self.layer1_a.t();
        debug_assert!(layer1_t.shape() == &[self.hidden_layer_size, 1], "layer1_t must be 100x1 matrix");
        let layer1_a_derived = layer1_t.mapv(ACTIVATION_PRIME);
        // delta_hidden: element-wise multiply of 100 x 1 and 100 x 1 results in 100 x 1 matrix.
        debug_assert!(weights2_by_delta_output.shape() == &[self.hidden_layer_size, 1], "weights2_by_delta_output must be 100x1 matrix");
        debug_assert!(layer1_a_derived.shape() == &[self.hidden_layer_size, 1], "layer1_a_derived must be 100x1 matrix");
        let delta_hidden: Array2<f32> =  weights2_by_delta_output * layer1_a_derived;

        // 4. Use the deltas to evaluate the derivatives
        // Error derivative with respect to weight2 (100x10 matrix).delta_output: 1x10. self.layer1: 1 x 100
        // dE / weight2[0][0] = delta_output[[0, 0]] * self.layer1[[0, 0]]
        // dE / weight2[0][1] = delta_output[[0, 1]] * self.layer1[[0, 0]]
        // ...
        // dE / weight2[0][9] = delta_output[[0, 9]] * self.layer1[[0, 0]]
        // dE / weight2[1][0] = delta_output[[0, 0]] * self.layer1[[0, 1]]
        // ...
        // dE / weight2[99][9]= delta_output[[0, 9]] * self.layer1[[0, 99]]

        // As weights2: 100 x 10, we want d_weights2 as 100 x 10. So transpose self.layer1: 1 x 100
        // For scalar-output, it was like the below with mapv
        // let d_weights2: Array2<f32> = self.layer1.t().mapv(|i| i * delta_output);
        // d_weights2: 100 x 10
        let d_weights2: Array2<f32> = self.layer1.t().dot(&delta_output);
        
        // As weights1: 3x4, we want d_weights1 as 3x4. delta_hidden: 4x1, last_input: 1x3
        // dE / weight1[0][0] = delta_hidden[[0, 0]] * self.last_input[[0, 0]]
        // dE / weight1[0][1] = delta_hidden[[1, 0]] * self.last_input[[0, 0]]
        // dE / weight1[0][2] = delta_hidden[[2, 0]] * self.last_input[[0, 0]]
        // dE / weight1[0][3] = delta_hidden[[3, 0]] * self.last_input[[0, 0]]
        // dE / weight1[1][0] = delta_hidden[[0, 0]] * self.last_input[[0, 1]]
        // ...
        // dE / weight1[2][3] = delta_hidden[[3, 0]] * self.last_input[[0, 2]]
        
        // Is this d_weights1 = delta_hidden.dot(&self.last_input) ?

        // delta_hidden: 4x1 multiplied by last_input: 1x3 = 4 x 3
        let delta_hidden_dot_input: Array2<f32> = delta_hidden.dot(self.last_input);
        let d_weights1: ArrayView2<f32> = delta_hidden_dot_input.t();

        self.weight2 += &(d_weights2.mapv(|i| i ));
        self.weight1 += &(d_weights1.mapv(|i| i ));
    }
}

type Grayscale = f32;


struct LabelTable {
    label_arrays: [Array2<f32>; 10]
}
impl LabelTable {
    pub fn new() -> LabelTable {
        let label_arrays: [Array2<f32>; 10] = [
            arr2(&[[1.,0.,0.,0.,0., 0.,0.,0.,0.,0.]]),
            arr2(&[[0.,1.,0.,0.,0., 0.,0.,0.,0.,0.]]),
            arr2(&[[0.,0.,1.,0.,0., 0.,0.,0.,0.,0.]]),
            arr2(&[[0.,0.,0.,1.,0., 0.,0.,0.,0.,0.]]),
            arr2(&[[0.,0.,0.,0.,1., 0.,0.,0.,0.,0.]]),
            arr2(&[[0.,0.,0.,0.,0., 1.,0.,0.,0.,0.]]),
            arr2(&[[0.,0.,0.,0.,0., 0.,1.,0.,0.,0.]]),
            arr2(&[[0.,0.,0.,0.,0., 0.,0.,1.,0.,0.]]),
            arr2(&[[0.,0.,0.,0.,0., 0.,0.,0.,1.,0.]]),
            arr2(&[[0.,0.,0.,0.,0., 0.,0.,0.,0.,1.]])
        ];
        LabelTable{label_arrays}
    }
    pub fn label_to_array<'a>(&'a self, num: i32) -> &'a Array2<f32> {
        &self.label_arrays[num as usize]
    }

    // array is 1x10 matrix
    pub fn array_to_label(array: ArrayView2<f32>) -> i32 {
        let mut max_value: f32 = 0.;
        let mut max_index: usize = 0;
        for i in 0..10 {
            if max_value < array[[0, i]] {
                max_value = array[[0, i]];
                max_index = i
            }
        }
        max_index as i32
    }
}

struct MnistRecord {
    label: i32,
    dots: [Grayscale; IMG_H_SIZE * IMG_W_SIZE],
    dots_array: Array2<f32>
}

impl MnistRecord {
    fn print(&self) {
        let mut s = String::new();
        let mut v: Vec<ANSIString<'static>> = Vec::new();
        for i in 0..IMG_H_SIZE {
            for j in 0..IMG_W_SIZE {
                if self.dots[i*IMG_H_SIZE + j] > 0. {
                    s.push_str("O");
                    let c = self.dots[i*IMG_H_SIZE + j];
                    debug_assert!(c >= 0. && c <= 1., "MNIST dot must be 0 - 255");
                    let p = (1. - c);
                    if p < 0. {
                        println!("p : {}", p);
                    }
                    let c256 = (p * MNIST_DOT_MAX) as u8;
                    let r = 255. - 232.;

                    // The bigger c is, the darker
                    let term_color = (255. - (r*c)) as u8;
                    debug_assert!(term_color >= 232, "xterm color range. u8 maximum is 255.");
                    let color = Fixed(term_color); // RGB(c256, c256, c256);
                    let t = Black.on(color).paint(" ");
                    v.push(t);
                } else {
                    s.push_str(" ");
                    let t = White.on(Fixed(231)).paint(" ");
                    v.push(t);
                }
            }
            s.push_str("\n");
            v.push(Style::default().paint("\n"));
        }
        // print!("Label {}:\n{}", self.label, s);
        print!("Image:\n{}",  ANSIStrings(&v));
    }
}



fn main() {
    let label_table = LabelTable::new();
    let file_path = "mnist_train.csv";
    let file_path_test = "mnist_test.csv";
    let file_test = File::open(file_path_test).unwrap();
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_reader(file);
    let mut rdr_test = csv::ReaderBuilder::new().has_headers(false).from_reader(file_test);
    let mut mnist_records:Vec<MnistRecord> = Vec::new();
    let mut mnist_records_test:Vec<MnistRecord> = Vec::new();
    let mut nn = NeuralNetwork::new(100);
    let before_record = Instant::now();
    for result in rdr_test.records() {
        let mut array: [Grayscale; IMG_W_SIZE * IMG_W_SIZE] = [0.; IMG_W_SIZE * IMG_W_SIZE];
        let record = result.unwrap();
        assert!(record.len() == 1+28*28);
        for i in 1..(IMG_W_SIZE * IMG_W_SIZE) {
            let v: f32 = record[i+1].parse().unwrap();
            let vv: f32 = v / MNIST_DOT_MAX;
            array[i] = vv;
            debug_assert!(vv >= 0. && vv <= 1., "MNIST dot must be between 0 - 255");
        }
        let label:i32 = record[0].parse().unwrap();
        let vv:Vec<f32> = array.to_vec();
//        let dots_array1: Array1<f32> = Array1::from_vec(vv);
        let dots_array2: Array2<f32> = Array2::from_shape_vec((1, 784), vv).unwrap();
        mnist_records_test.push(MnistRecord{
            label,
            dots: array,
            dots_array: dots_array2
        });
    }
    println!("Read {} MNIST test records in {} secs", mnist_records_test.len(), before_record.elapsed().as_secs());
    for result in rdr.records() {
        let mut array: [Grayscale; IMG_W_SIZE * IMG_W_SIZE] = [0.; IMG_W_SIZE * IMG_W_SIZE];
        let record = result.unwrap();
        assert!(record.len() == 1+28*28);
        for i in 1..(IMG_W_SIZE * IMG_W_SIZE) {
            let v: f32 = record[i+1].parse().unwrap();
            let vv: f32 = v / MNIST_DOT_MAX;
            array[i] = vv;
            debug_assert!(vv >= 0. && vv <= 1., "MNIST dot must be between 0 - 255");
        }
        let label:i32 = record[0].parse().unwrap();
        let vv:Vec<f32> = array.to_vec();
//        let dots_array1: Array1<f32> = Array1::from_vec(vv);
        let dots_array2: Array2<f32> = Array2::from_shape_vec((1, 784), vv).unwrap();
        let mnist: MnistRecord = MnistRecord{
            label,
            dots: array,
            dots_array: dots_array2
        };
        // let y = label_table.label_to_array(mnist.label);
        // nn.feed_forward(&mnist.dots_array);
        // nn.back_prop(y);
        // mnist.print();
        mnist_records.push(mnist);
    }
    println!("Read {} MNIST records in {} secs", mnist_records.len(), before_record.elapsed().as_secs());
    println!("Starting training of {} hidden layer", nn.hidden_layer_size);
    let before_training = Instant::now();
    let epoch = 30;
    for i in 0..epoch {
        let mut total_count = 0;
        let mut correct_prediction = 0;
        for mnist in mnist_records.iter() {
            let y = label_table.label_to_array(mnist.label);
            let predicted_label = nn.feed_forward(&mnist.dots_array);
            nn.back_prop(y);
            if predicted_label == mnist.label {
                correct_prediction += 1;
            }
            total_count += 1;
        }
        for i in 0..3 {
            let sample_index = thread_rng().gen_range(0, mnist_records.len());
            let mnist_sample = &mnist_records[sample_index as usize];
            mnist_sample.print();
            let predicted_label = nn.feed_forward(&mnist_sample.dots_array);
            println!("Predicted: {}\n--------------------------------", predicted_label);
        }
        println!("Finished epoch {}. Prediction rate: {}",
            i,  (correct_prediction as f32) / (total_count as f32));
    }
    println!("Finished training in {} secs", before_training.elapsed().as_secs());
    let mut test_count = 0;
    let mut test_correct_prediction = 0;
    for mnist in mnist_records_test.iter() {
        let predicted_label = nn.feed_forward(&mnist.dots_array);
        if predicted_label == mnist.label {
            test_correct_prediction += 1;
        }
        test_count += 1;
    }
    println!("Test result: {} of {} test cases",
     (test_correct_prediction as f32) / (test_count as f32), test_count);
}



#[test]
fn array_label_test() {
    for i in 0..10 {
        let lt = LabelTable::new();
        let array = lt.label_to_array(i);
        let actual_label = LabelTable::array_to_label(array.view());
        assert_eq!(actual_label, i, "The array to label conversion should work for i")
    }
}


#[test]
fn backprop_test() {
    let label_table = LabelTable::new();
    let file_path = "mnist_train.csv";
    let file = File::open(file_path).unwrap();
    let mnist_csv_reader_builder = csv::ReaderBuilder::new().has_headers(false);
    let mut rdr = mnist_csv_reader_builder.from_reader(file);

    let mut mnist_records:Vec<MnistRecord> = Vec::new();
    let mut nn = NeuralNetwork::new(100);
    for result in rdr.records() {
        let mut array: [Grayscale; IMG_W_SIZE * IMG_W_SIZE] = [0.; IMG_W_SIZE * IMG_W_SIZE];
        let record = result.unwrap();
        assert!(record.len() == 1+28*28);
        for i in 1..(IMG_W_SIZE * IMG_W_SIZE) {
            let v: f32 = record[i+1].parse().unwrap();
            array[i] = v / 255.;
        }
        let label:i32 = record[0].parse().unwrap();
        let vv:Vec<f32> = array.to_vec();
//        let dots_array1: Array1<f32> = Array1::from_vec(vv);
        let dots_array2: Array2<f32> = Array2::from_shape_vec((1, 784), vv).unwrap();
        let mnist: MnistRecord = MnistRecord{
            label,
            dots: array,
            dots_array: dots_array2
        };
        // let y = label_table.label_to_array(mnist.label);
        // nn.feed_forward(&mnist.dots_array);
        // nn.back_prop(y);
        mnist_records.push(mnist);
        break; // for testing
    }
    let epoch = 10;
    for i in 0..epoch {
        for mnist in mnist_records.iter() {
            let y = label_table.label_to_array(mnist.label);
            nn.feed_forward(&mnist.dots_array);
            nn.back_prop(y);
        }
    }
}


