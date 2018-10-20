// Implementation of Convolutional Neural Network for MNIST
// The implementation is from
//   ゼロから作るDeep Learning ――Pythonで学ぶディープラーニングの理論と実装
// Sample Code https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch07/simple_convnet.py in Python
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
use layer::{Affine, Convolution, Elem, Layer, Matrix, Relu, SoftmaxWithLoss};
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
        let t = rng.gen_range(0, mnist_record.len());
        let mut assign_mut = ret.slice_mut(s![i, 0, .., ..]);
        let record = &mnist_record[t];
        assign_mut.assign(&record.dots_array);
        answer_labels.push(record.label);
    }
    (ret, answer_labels)
}

fn main() {
    let mnist_records_train: Vec<MnistRecord> =
        MnistRecord::load_from_csv("mnist_train.csv").unwrap();
    let mnist_records_test: Vec<MnistRecord> =
        MnistRecord::load_from_csv("mnist_test.csv").unwrap();
    let (nchw, answers) = generate_conv_input_array4(&mnist_records_train, 10);
    let answer_array1 = Array1::from_vec(answers);

    // Initialize layers
    let mut convolution_layer = Convolution::new(10, 30, 3, 5, 5, 1, 0);
    let mut affine_layer = Affine::new(100, 10);
    let mut relu_layer = Relu::<Ix4>::new();
    let mut relu2_layer = Relu::<Ix2>::new();
    let mut affine2_layer = Affine::new(100, 10);
    let mut softmax_layer = SoftmaxWithLoss::new();

    let label_table = LabelTable::new();
    let before_training = Instant::now();
    let epoch = 30;
    for i in 0..epoch {
        let mut total_count = 0;
        let mut correct_prediction = 0;

        // 4-dimensional data of (N-data, Channel, Height, Width)
        {
            // Forward

            // nchw: borrowed value does not live long enough
            // hchw is used only to train the internal values of the layer
            let conv_output = convolution_layer.forward(&nchw);
            let relu_output = relu_layer.forward(&conv_output);
            let affine_output = affine_layer.forward(&relu_output);
            let relu2_output = relu2_layer.forward(&affine_output);
            //let affine2_output = affine2_layer.forward(&relu2_output);

            let softmax_output = softmax_layer.forward(&relu2_output, &answer_array1);

            // Backward?
            println!(
                "Finished epoch {}. Prediction rate: {}",
                i,
                (correct_prediction as f32) / (total_count as f32)
            );
        }
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


struct S<'a> {
    counter: i32,
    by: &'a i32
}
impl<'a> S<'a> {
    pub fn increment(&mut self) {
        self.counter  += 1;
    }
    pub fn increment_by(&mut self, inc: &'a i32) {
        self.counter += *inc;
        self.by = inc;
    }
}

#[test]
fn test_struct_reference() {
    let mut s = S{ counter: 0 };
    for i in 0..10 {
        let k = i + 5;
        s.increment_by(&k);
        println!("i = {}", i);

    }
    println!("S.counter = {}", s.counter);
}