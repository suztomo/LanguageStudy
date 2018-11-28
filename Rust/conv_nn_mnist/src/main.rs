// Implementation of Convolutional Neural Network for MNIST
// The implementation is from
//   ゼロから作るDeep Learning ――Pythonで学ぶディープラーニングの理論と実装
// Sample Code https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch07/simple_convnet.py in Python
extern crate csv;
use std::fs::File;
use std::time::Instant;

#[macro_use]
extern crate assert_approx_eq;

extern crate ansi_term;

#[macro_use]
extern crate lazy_static; // 1.0.1;

extern crate num_traits;

#[macro_use]
extern crate ndarray;
//extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate rand;

use rand::Rng;
// use rand::Rng;
use ndarray::prelude::*;
extern crate utils;

mod layer;
use layer::{argmax2d, Affine, Convolution, Elem, Layer, Matrix, Pooling, Relu, SoftmaxWithLoss};
mod mnist;
use mnist::{Grayscale, MnistRecord, IMG_H_SIZE, IMG_W_SIZE};

mod gradient;

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
    pub fn label_to_array<'a>(&'a self, num: usize) -> &'a Array2<f32> {
        &self.label_arrays[num]
    }

    // array is 1x10 matrix
    pub fn array_to_label(array: ArrayView2<f32>) -> usize {
        let mut max_value: f32 = 0.;
        let mut max_index: usize = 0;
        for i in 0..10 {
            if max_value < array[[0, i]] {
                max_value = array[[0, i]];
                max_index = i
            }
        }
        max_index as usize
    }
}

/**
 * Given vector of mnist records and input_size, this returns
 * the Array4 of (N, C, H, W) for Convolutional Neural Network input
 * and the labels of N
 */
fn generate_conv_input_array4(
    mnist_records: &Vec<MnistRecord>,
    n_input: usize,
) -> (Matrix, Vec<usize>) {
    let channel_count = 1; // MNIST is grayscale
    let mut rng = rand::thread_rng();
    let mut answer_labels = Vec::<usize>::new();
    let mut ret: Array4<Elem> =
        Array4::<Elem>::zeros((n_input, channel_count, IMG_H_SIZE, IMG_W_SIZE));
    for i in 0..n_input {
        let t = rng.gen_range(0, mnist_records.len());
        let mut assign_mut = ret.slice_mut(s![i, 0, .., ..]);
        let record = &mnist_records[t];
        assign_mut.assign(&record.dots_array.mapv(f64::from));
        answer_labels.push(record.label);
    }
    (ret, answer_labels)
}

fn mnist_to_nchw(mnist_record: &MnistRecord) -> Matrix {
    let mut ret: Array4<Elem> = Array4::<Elem>::zeros((1, 1, IMG_H_SIZE, IMG_W_SIZE));
    {
        let mut assign_mut = ret.slice_mut(s![0, 0, .., ..]);
        assign_mut.assign(&mnist_record.dots_array.mapv(f64::from));
    }
    ret
}

fn main() {
    let mnist_records_train: Vec<MnistRecord> =
        MnistRecord::load_from_csv("mnist_train.csv").unwrap();
    let mnist_records_test: Vec<MnistRecord> =
        MnistRecord::load_from_csv("mnist_test.csv").unwrap();

    // Initialize layers
    // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch07/simple_convnet.py
    let padding = 0; // To make 24x24 to 28x28
    let batch_size = 100;
    let mut convolution_layer = Convolution::new(batch_size, 30, 1, 5, 5, 1, padding);
    let mut relu_layer = Relu::<Ix4>::new();
    let mut pooling_layer = Pooling::new(2, 2, 2, 0);
    let affine_output_size = 100;
    let mut affine_layer = Affine::new(4320, affine_output_size);
    let mut relu2_layer = Relu::<Ix2>::new();
    let mut affine2_layer = Affine::new(affine_output_size, 10);
    let mut softmax_layer = SoftmaxWithLoss::new();

    let before_training = Instant::now();
    let epoch = 300;
    let learning_rate = 0.1;

    // This needs to be in the for-loop. However, even when this is outside the loop,
    // the learning rate (softmax_output) does not improve over iteration. Why?

    for i in 0..epoch {
        // This 10 must match the first argument for Convolution::new.
        let (nchw, answers) = generate_conv_input_array4(&mnist_records_train, batch_size);
        //println!("answers: {:?}", answers);
        let answer_array1 = Array1::from_vec(answers);

        // nchw is 4-dimensional data of (N-data, Channel, Height, Width)
        // Forward

        // nchw: borrowed value does not live long enough
        // hchw is used only to train the internal values of the layer
        if i == 0 {
            println!("nchw shape: {:?}", nchw.shape());
        }
        let conv_output = convolution_layer.forward(&nchw);
        if i == 0 {
            println!("conv_output shape: {:?}", conv_output.shape());
        }
        let relu_output = relu_layer.forward(&conv_output);
        if i == 0 {
            println!("relu_output shape: {:?}", relu_output.shape());
        }
        let pooling_output = pooling_layer.forward(&relu_output);
        if i == 0 {
            println!("pooling_output shape: {:?}", pooling_output.shape());
        }
        let affine_output = affine_layer.forward(&pooling_output);
        if i == 0 {
            println!("affine_output shape: {:?}", affine_output.shape());
        }
        let relu2_output = relu2_layer.forward(&affine_output);
        if i == 0 {
            println!("relu2_output shape: {:?}", relu2_output.shape());
        }

        let affine2_output = affine2_layer.forward_2d(&relu2_output);
        if i == 0 {
            println!("affine2_output shape: {:?}", affine2_output.shape());
        }
        //        println!("affine2_output: {:?}", affine2_output);

        let softmax_output = softmax_layer.forward(&affine2_output, &answer_array1);

        // It always stick to 3.3219. Why? It looks like the all elements of affine2_output
        // are close to zero. The feedback is not working as expected.
        // As of Nov 29th, it always stick to ~2.30 after fixing softmax.forward
        println!(
            "Finished epoch {}. softmax_output (smaller, the better): {}",
            i, softmax_output
        );

        // Backward
        // Somehow backpropagation of softmax always takes 1.0.
        // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch07/simple_convnet.py#L129
        // This is because softmax_with_loss.backward discards the input value.
        // The returned value from backward only depends on the layer's internal state.
        let softmax_dx = softmax_layer.backward(1.);
        let affine2_dx = affine2_layer.backward_2d(&softmax_dx);
        affine2_layer.weights -= &(&affine2_layer.d_weights * learning_rate);
        affine2_layer.bias -= &(&affine2_layer.d_bias * learning_rate);

        let relu2_dx = relu2_layer.backward(&affine2_dx);
        let affine_dx = affine_layer.backward(&relu2_dx);
        affine_layer.weights -= &(&affine_layer.d_weights * learning_rate);
        affine_layer.bias -= &(&affine_layer.d_bias * learning_rate);
        let pooling_dx = pooling_layer.backward(&affine_dx);
        let relu_dx = relu_layer.backward(&pooling_dx);
        let _conv_dx = convolution_layer.backward(&relu_dx);
        convolution_layer.weights -= &(&convolution_layer.d_weights * learning_rate);
        convolution_layer.bias -= &(&convolution_layer.d_bias * learning_rate);
    }
    println!(
        "Finished training set in {} secs. Running in test set.",
        before_training.elapsed().as_secs()
    );
    let mut test_count = 0;
    let mut test_correct_prediction = 0;
    for mnist in mnist_records_test.iter() {
        let nchw = mnist_to_nchw(&mnist);
        let conv_output = convolution_layer.forward(&nchw);
        let relu_output = relu_layer.forward(&conv_output);
        let pooling_output = pooling_layer.forward(&relu_output);
        let affine_output = affine_layer.forward(&pooling_output);
        let relu2_output = relu2_layer.forward(&affine_output);
        let affine2_output = affine2_layer.forward_2d(&relu2_output);
        let affine2_output_argmax = argmax2d(&affine2_output, Axis(1));
        let predicted_label = affine2_output_argmax[[0]];
        if predicted_label == mnist.label {
            test_correct_prediction += 1;
        }
        test_count += 1;
    }
    println!(
        "Test result: {} of {} test cases",
        (test_correct_prediction as Elem) / (test_count as Elem),
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
            actual_label, i as usize,
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
        let label: usize = record[0].parse().unwrap();
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
    for _i in 0..epoch {
        for mnist in mnist_records.iter() {
            let _y = label_table.label_to_array(mnist.label);
            //nn.feed_forward(&mnist.dots_array);
            //nn.back_prop(y);
        }
    }
}

#[test]
fn test_generate_conv_input_array4() {
    let mnist_train_data_res = MnistRecord::load_from_csv("tests/mnist_test_10.csv");
    let mnist_train_data: Vec<MnistRecord> = mnist_train_data_res.unwrap();
    let (m, _answers) = generate_conv_input_array4(&mnist_train_data, 10);
    assert_eq!(
        m.shape(),
        &[10, 1, 28, 28],
        "10 input, channel 1 (grayscale), width: 28 and height:28"
    );
}
