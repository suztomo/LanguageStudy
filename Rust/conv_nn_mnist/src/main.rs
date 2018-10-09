// Implementation of Convolutional Neural Network for MNIST
// The implementation is from
//   ゼロから作るDeep Learning ――Pythonで学ぶディープラーニングの理論と実装
// Sample Code https://github.com/oreilly-japan/deep-learning-from-scratch in Python
extern crate csv;
use std::fs::File;
use std::time::Instant;
use std::{thread, time};

#[macro_use]
extern crate assert_approx_eq;

extern crate ansi_term;

#[macro_use]
extern crate lazy_static; // 1.0.1;

#[macro_use]
extern crate ndarray;
//extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate rand;

use ndarray_rand::{RandomExt, F32};
use rand::distributions::{IndependentSample, Normal, Range};
use rand::{thread_rng, Rng};
// use rand::Rng;
use ndarray::prelude::*;
extern crate utils;
use utils::math::sigmoid;

mod layer;
use layer::{Affine, Convolution, Elem, Layer, Matrix, Relu};
mod mnist;
use mnist::{Grayscale, MnistRecord, IMG_H_SIZE, IMG_W_SIZE};

fn sigmoid_derivative(x: f32) -> f32 {
    // https://beckernick.github.io/sigmoid-derivative-neural-network/
    let t = sigmoid(x);
    t * (1.0 - t)
}

const ACTIVATION: &Fn(f32) -> f32 = &sigmoid;
const ACTIVATION_PRIME: &Fn(f32) -> f32 = &sigmoid_derivative;

fn activation_array(input: &Array2<f32>) -> Array2<f32> {
    input.mapv(ACTIVATION)
}

// Tanh doesn't work as the range is -1 to + 1
//const ACTIVATION:&Fn(f32) -> f32 = &tanh;
//const ACTIVATION_PRIME: &Fn(f32) -> f32 = &tanh_prime;

#[derive(Debug)]
struct NeuralNetwork<'a> {
    hidden_layer_size: usize,
    // <row count> x <column count>
    last_input: &'a Array2<f32>, // Input 1 x 784 (28x28)
    weight1: Array2<f32>,        // 784 x 100
    layer1_a: Array2<f32>,       // 1 x 100, before activation
    layer1: Array2<f32>,         // 1 x 100
    weight2: Array2<f32>,        // 100 x 10
    last_output_a: Array2<f32>, // 1 x 10. before activation. PRML uses 'a' for before activation. 'z' for after activation.
    last_output: Array2<f32>,   // 1 x 10 // one-hot representation for 10 digits (0-9)
}

lazy_static! {
    static ref INPUT_ZERO: Array2<f32> = Array::zeros((1, IMG_H_SIZE * IMG_W_SIZE));
    static ref INPUT_ARRAY4_ZERO: Matrix = Array::zeros((1, 1, 1, 1));
}

struct LabelTable {
    label_arrays: [Array2<f32>; 10],
}
impl LabelTable {
    pub fn new() -> LabelTable {
        let label_arrays: [Array2<f32>; 10] = [
            arr2(&[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
            arr2(&[[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]]),
            arr2(&[[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]]),
            arr2(&[[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]]),
            arr2(&[[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]]),
            arr2(&[[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]),
            arr2(&[[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]),
            arr2(&[[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]]),
            arr2(&[[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]]),
            arr2(&[[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]),
        ];
        LabelTable { label_arrays }
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

/**
 * Given vector of mnist records and input_size, this returns
 * the Array4 of (N, C, H, W) for Convolutional Neural Network input
 * and the labels of N
 */
fn generate_conv_input_array4(
    mnist_record: &Vec<MnistRecord>,
    n_input: usize,
) -> (Matrix, Vec<i32>) {
    let channel_count = 1; // MNIST is grayscale
    let mut rng = rand::thread_rng();
    let mut answer_labels = Vec::<i32>::new();
    let mut ret: Array4<Elem> =
        Array4::<Elem>::zeros((n_input, channel_count, IMG_H_SIZE, IMG_W_SIZE));
    for i in 0..n_input {
        let t = thread_rng().gen_range(0, mnist_record.len());
        let mut assign_mut = ret.slice_mut(s![i, 0, .., ..]);
        let record = &mnist_record[t];
        assign_mut.assign(&record.dots_array);
        answer_labels.push(record.label);
    }
    (ret, answer_labels)
}

fn main() {
    // Initialize layers
    let input = Array::zeros((1, 1, 1, 1));
    let mut convolution_layer = Convolution::new(10, 30, 3, 5, 5, 1, 0);
    let mut affine_layer = Affine::new(100, 10);
    let mut relu_layer = Relu::<Ix4>::new();
    let mut relu2_layer = Relu::<Ix2>::new();

    let conv_output = convolution_layer.forward(&input);
    let relu_output = relu_layer.forward(&conv_output);
    let affine_output = affine_layer.forward(&relu_output);
    let relu2_output = relu2_layer.forward(&affine_output);
    // Because relu2_layer and affine are not compliant with Layer (Array4)
    // we cannot create vector for layers... how can we?
    let layer_vec: Vec<&Layer> = vec![&convolution_layer, &relu_layer];
    // let layer_vec: Vec<&Layer> = vec![&convolution_layer, &relu_layer, &affine_layer];
    println!("layer array: {:?}", layer_vec.len());

    let label_table = LabelTable::new();
    let mnist_records_train: Vec<MnistRecord> =
        MnistRecord::load_from_csv("mnist_train.csv").unwrap();
    let mnist_records_test: Vec<MnistRecord> =
        MnistRecord::load_from_csv("mnist_test.csv").unwrap();
    let before_training = Instant::now();
    let epoch = 30;
    for i in 0..epoch {
        let mut total_count = 0;
        let mut correct_prediction = 0;
        for mnist in mnist_records_train.iter() {
            let y = label_table.label_to_array(mnist.label);
            // dummy
            let predicted_label = 1;
            if predicted_label == mnist.label {
                correct_prediction += 1;
            }
            total_count += 1;
        }
        for i in 0..3 {
            let sample_index = thread_rng().gen_range(0, mnist_records_train.len());
            let mnist_sample = &mnist_records_train[sample_index as usize];
            mnist_sample.print();
        }
        println!(
            "Finished epoch {}. Prediction rate: {}",
            i,
            (correct_prediction as f32) / (total_count as f32)
        );
    }
    println!(
        "Finished training in {} secs",
        before_training.elapsed().as_secs()
    );
    let mut test_count = 0;
    let mut test_correct_prediction = 0;
    for mnist in mnist_records_test.iter() {
        let predicted_label = 1; // nn.feed_forward(&mnist.dots_array);
        if predicted_label == mnist.label {
            test_correct_prediction += 1;
        }
        test_count += 1;
    }
    println!(
        "Test result: {} of {} test cases",
        (test_correct_prediction as f32) / (test_count as f32),
        test_count
    );
}

#[test]
fn array_label_test() {
    for i in 0..10 {
        let lt = LabelTable::new();
        let array = lt.label_to_array(i);
        let actual_label = LabelTable::array_to_label(array.view());
        assert_eq!(
            actual_label, i,
            "The array to label conversion should work for i"
        )
    }
}

#[test]
fn backprop_test() {
    let label_table = LabelTable::new();
    let file_path = "mnist_test.csv";
    let file = File::open(file_path).unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    let mut mnist_records: Vec<MnistRecord> = Vec::new();
    for result in rdr.records() {
        let mut array: [Grayscale; IMG_W_SIZE * IMG_W_SIZE] = [0.; IMG_W_SIZE * IMG_W_SIZE];
        let record = result.unwrap();
        assert!(record.len() == 1 + 28 * 28);
        for i in 1..(IMG_W_SIZE * IMG_W_SIZE) {
            let v: f32 = record[i + 1].parse().unwrap();
            array[i] = v / 255.;
        }
        let label: i32 = record[0].parse().unwrap();
        let vv: Vec<f32> = array.to_vec();
        //        let dots_array1: Array1<f32> = Array1::from_vec(vv);
        let dots_array2: Array2<f32> = Array2::from_shape_vec((1, 784), vv).unwrap();
        let mnist: MnistRecord = MnistRecord {
            label,
            dots: array,
            dots_array: dots_array2,
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
            //nn.feed_forward(&mnist.dots_array);
            //nn.back_prop(y);
        }
    }
}

#[test]
fn test_generate_conv_input_array4() {
    let mnist_train_data_res = MnistRecord::load_from_csv("tests/mnist_test_10.csv");
    let mnist_train_data: Vec<MnistRecord> = mnist_train_data_res.unwrap();
    let (m, answers) = generate_conv_input_array4(&mnist_train_data, 10);
    assert_eq!(
        m.shape(),
        &[10, 1, 28, 28],
        "10 input, channel 1 (grayscale), width: 28 and height:28"
    );
}
