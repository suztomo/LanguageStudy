#[macro_use]
extern crate ndarray;
//extern crate ndarray_linalg;
extern crate rand;
use rand::Rng;
use ndarray::prelude::*;
extern crate utils;
use utils::math::sigmoid;

// Rust implementation of Neural Network learning:
// https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

// NDArray for Numpy users
// https://docs.rs/ndarray/0.11.2/ndarray/doc/ndarray_for_numpy_users/index.html

// Handwritten digit recognizer
// http://handwritten-digits-recognizer.herokuapp.com/

// Let's say input is (a:f32, b:f32, c:f32). This is Array1<f32>
#[derive(Debug)]
struct NeuralNetwork{
    // Input 1 x 3
    lastInput: Array1<f32>,
    weight1: Array2<f32>, // 3 x 4
    weight2: Array1<f32>, // 4 x 1
}

impl NeuralNetwork {
    fn feed_forward(&mut self, input: Array1<f32>) -> f32 {
        self.lastInput = input;
        //  'ndarray: inputs 3 × 2 and 3 × 1 are not compatible for matrix multiplication'
        let dot_product:Array1<f32> = self.lastInput.dot(&self.weight1);
        // https://docs.rs/ndarray/0.11.2/ndarray/struct.ArrayBase.html#method.mapv
        let layer1:Array1<f32> = dot_product.mapv(sigmoid);

        // When the matrix size is different:
        //   'assertion failed: `(left == right)`
        //     left: `4`,
        //     right: `3`'
        let output:f32 = layer1.dot(&self.weight2);
        output
    }
}

fn main() {
    let kookoo = 0;
    let m = kookoo;
    println!("foo");

    let init_weight1 = arr2(
        &[[0.1, -0.8, 1.1, 0.3],
        [0.6, -0.2, 0.5, 1.0],
        [0.6, -0.2, 0.5, -0.4]]
    );
    let init_weight2 = arr1(&[0.23, 0.21, -0.5, 0.1]);
    let mut nn = NeuralNetwork {
        lastInput: arr1(&[]),
        weight1 : init_weight1,
        weight2 : init_weight2,
    };
    let input1 = arr1(&[0.23, 0.21, -0.5]);
    let res1 = nn.feed_forward(input1);
    println!("Initialized NN {:?}, {}", nn, res1);

    let training_data: Array2<f32> = arr2(&[
        [0., 0., 1.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.]
        ]);
    let ans = vec![0., 0., 0., 1.];
    println!("training data {:?} {:?}", training_data, ans);
    
}
