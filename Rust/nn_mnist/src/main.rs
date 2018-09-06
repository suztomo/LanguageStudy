// Implementation of Neural Network for MNIST
// Python implementation: https://www.python-course.eu/neural_network_mnist.php
extern crate csv;
use std::fs::File;
use std::{thread, time};

#[macro_use]
extern crate ndarray;
//extern crate ndarray_linalg;
extern crate rand;
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

fn bad_sigmoid_derivative(x: f32) -> f32 {
    // https://iamtrask.github.io/2015/07/12/basic-python-network/
    x * (1.0 - x)
}


// This works with back_prop_prml
const ACTIVATION:&Fn(f32) -> f32 = &sigmoid;
const ACTIVATION_PRIME: &Fn(f32) -> f32 = &sigmoid_derivative;
// This works with back_prop_
//const ACTIVATION:&Fn(f32) -> f32 = &sigmoid;
//const ACTIVATION_PRIME: &Fn(f32) -> f32 = &bad_sigmoid_derivative;
//const ACTIVATION:&Fn(f32) -> f32 = &tanh;
//const ACTIVATION_PRIME: &Fn(f32) -> f32 = &tanh_prime;

#[derive(Debug)]
struct NeuralNetwork{
    // <row count> x <column count>
    last_input: Array2<f32>, // Input 1 x 784 (28x28)
    weight1: Array2<f32>,    // 784 x 100
    layer1_a: Array2<f32>,   // 1 x 100, before activation
    layer1: Array2<f32>,     // 1 x 100
    weight2: Array2<f32>,    // 100 x 10
    last_output_a: Array2<f32>, // 1 x 10. before activation. PRML uses 'a' for before activation. 'z' for after activation.
    last_output: Array2<f32>, // 1 x 10 // one-hot representation for 10 digits (0-9)
}

impl NeuralNetwork {
    fn feed_forward(&mut self, input_array: ArrayView1<f32>) -> f32 {
        self.last_input = input_array.to_owned().insert_axis(Axis(0));
        // When theres' wrong number of rows and column, you'll get following runtime error:
        //   'ndarray: inputs 3 × 2 and 3 × 1 are not compatible for matrix multiplication'

        // lastInput: 1 x 3, weight1: 3 x 4, and thus dot_product: 1 x 4
        self.layer1_a = self.last_input.dot(&self.weight1);
        // https://docs.rs/ndarray/0.11.2/ndarray/struct.ArrayBase.html#method.mapv
        // layer1: 1 x 4
        self.layer1 = self.layer1_a.mapv(ACTIVATION);
 
        // When the matrix size is different:
        //   'assertion failed: `(left == right)`
        //     left: `4`,
        //     right: `3`'
        // layer1: 1 x 4, weight2: 4 x 1, and thus output is 1 x 1 (Array2)
        let output_matrix:Array2<f32> = self.layer1.dot(&self.weight2);
        // before activation
        self.last_output_a = output_matrix;
        self.last_output = self.last_output_a.mapv(ACTIVATION);
        self.last_output[[0,0]]
    }

    fn back_prop(&mut self, y: Array2<f32>) {
        // Input layer     hidden layer       output layer
        //   3 -> (weights1) -> 4 -> (weights2) -> 1

        // PRML p.244 "Error Backpropagation"
        // 1. "Apply an input vector..." is already done at feedforward function

        // 2. Evaluate δ (delta) for all hte output unit. In this case it's only one value
        // Get Error δ (delta) at the output, the output layer only has one element so f32 is enough.
        let delta_output:Array2<f32> = y - &self.last_output;


        // 3. Backpropagate deltas using next delta and weight
        // In this case, hidden Layer has 4 nodes. So the error (delta) is 4-element matrix
        //     self.layer1_a : 1 x 4 matrix
        // hidden_delta_0 = h'(self.layer1_a[[0, 0]]) * (weights2[[0, 0]] * delta_output)
        // hidden_delta_1 = h'(self.layer1_a[[0, 1]]) * (weights2[[1, 0]] * delta_output)
        // hidden_delta_2 = h'(self.layer1_a[[0, 2]]) * (weights2[[2, 0]] * delta_output)
        // hidden_delta_3 = h'(self.layer1_a[[0, 3]]) * (weights2[[3, 0]] * delta_output)

        // self.weight2 is 4x1 matrix, so as weights2_by_delta_output
        // For scalar-output, it was like the below with mapv
        // let weights2_by_delta_output: Array2<f32> = self.weight2.mapv(|i| i * delta_output);
        let weights2_by_delta_output: Array2<f32> = self.weight2.dot(&delta_output);
        // 4x1 matrix element-wise multiplied by 4x1 (transpose of 1x4) matrix
        // delta_hidden: 4x1
        let layer1_a_derived = self.layer1_a.t().mapv(ACTIVATION_PRIME);
        let delta_hidden: Array2<f32> =  weights2_by_delta_output * layer1_a_derived;


        // 4. Use the deltas to evaluate the derivatives
        // Error derivative with respect to weight2 (4x1 matrix). self.layer1: 1 x 4
        // dE / weight2[0][0] = delta_output * self.layer1[[0, 0]]
        // dE / weight2[1][0] = delta_output * self.layer1[[0, 1]]
        // dE / weight2[2][0] = delta_output * self.layer1[[0, 2]]
        // dE / weight2[3][0] = delta_output * self.layer1[[0, 3]]

        // As weights2: 4x1, we want d_weights2 as 4x1. So transpose self.layer1
        // For scalar-output, it was like the below with mapv
        // let d_weights2: Array2<f32> = self.layer1.t().mapv(|i| i * delta_output);
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
        let delta_hidden_dot_input: Array2<f32> = delta_hidden.dot(&self.last_input);
        let d_weights1: ArrayView2<f32> = delta_hidden_dot_input.t();

        self.weight2 += &(d_weights2.mapv(|i| i ));
        self.weight1 += &(d_weights1.mapv(|i| i ));
    }
}

const IMG_H_SIZE: usize = 28;
const IMG_W_SIZE: usize = 28;

type Grayscale = i32;

struct MnistRecord {
    label: i32,
    dots: [Grayscale; IMG_H_SIZE * IMG_W_SIZE]
}

impl MnistRecord {
    fn print(&self) {
        let mut s = String::new();
        for i in 0..IMG_H_SIZE {
            for j in 0..IMG_W_SIZE {
                if self.dots[i*IMG_H_SIZE + j] > 0 {
                    s.push_str("o");
                } else {
                    s.push_str(" ");
                }
            }
            s.push_str("\n");
        }
        print!("Label {}:\n{}", self.label, s);
    }
}

fn main() {
    let file_path = "mnist_train.csv";
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let mut array: [i32; IMG_W_SIZE * IMG_W_SIZE] = [0; IMG_W_SIZE * IMG_W_SIZE];
        let record = result.unwrap();
        assert!(record.len() == 1+28*28);
        for i in 1..(IMG_W_SIZE * IMG_W_SIZE) {
            array[i] = record[i+1].parse().unwrap();
        }
        let label:i32 = record[0].parse().unwrap();
        let mnist: MnistRecord = MnistRecord{
            label,
            dots: array,
        };
        mnist.print();
        thread::sleep(time::Duration::from_millis(400));
    }
}

