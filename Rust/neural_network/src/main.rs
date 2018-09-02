#[macro_use]
extern crate ndarray;
//extern crate ndarray_linalg;
extern crate rand;
// use rand::Rng;
use ndarray::prelude::*;
extern crate utils;
use utils::math::sigmoid;


// Rust implementation of Neural Network learning:
// https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

// NDArray for Numpy users
// https://docs.rs/ndarray/0.11.2/ndarray/doc/ndarray_for_numpy_users/index.html

// Handwritten digit recognizer
// http://handwritten-digits-recognizer.herokuapp.com/

// Youtube video https://www.youtube.com/watch?v=tIeHLnjs5U8
// Backpropagation calculus | Deep learning, chapter 4

// Let's say input is (a:f32, b:f32, c:f32). This is Array1<f32>
#[derive(Debug)]
struct NeuralNetwork{
    // <row count> x <column count>
    last_input: Array2<f32>, // Input 1 x 3
    weight1: Array2<f32>,   // 3 x 4
    layer1_a: Array2<f32>,   // 1 x 4, before activation
    layer1: Array2<f32>,    // 1 x 4
    weight2: Array2<f32>,   // 4 x 1
    last_output_a: f32,     // before activation
    last_output: f32, // scalar
}

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
        self.last_output_a = output_matrix[[0, 0]];
        self.last_output = ACTIVATION(self.last_output_a);
        self.last_output
    }

    fn back_prod_prml(&mut self, y: f32) {
        // Input layer     hidden layer       output layer
        //   3 -> (weights1) -> 4 -> (weights2) -> 1

        // PRML p.244 "Error Backpropagation"
        // 1. "Apply an input vector..." is already done at feedforward function

        // 2. Evaluate δ (delta) for all hte output unit. In this case it's only one value
        // Get Error δ (delta) at the output, the output layer only has one element so f32 is enough.
        let delta_output:f32 = y - self.last_output;


        // 3. Backpropagate deltas using next delta and weight
        // In this case, hidden Layer has 4 nodes. So the error (delta) is 4-element matrix
        //     self.layer1_a : 1 x 4 matrix
        // hidden_delta_0 = h'(self.layer1_a[[0, 0]]) * (weights2[[0, 0]] * delta_output)
        // hidden_delta_1 = h'(self.layer1_a[[0, 1]]) * (weights2[[1, 0]] * delta_output)
        // hidden_delta_2 = h'(self.layer1_a[[0, 2]]) * (weights2[[2, 0]] * delta_output)
        // hidden_delta_3 = h'(self.layer1_a[[0, 3]]) * (weights2[[3, 0]] * delta_output)

        // self.weight2 is 4x1 matrix, so as weights2_by_delta_output
        let weights2_by_delta_output: Array2<f32> = self.weight2.mapv(|i| i * delta_output);
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
        let d_weights2: Array2<f32> = self.layer1.t().mapv(|i| i * delta_output);

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

    fn back_prop(&mut self, y: f32) {
        // d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        // d_weights1 = np.dot(self.input.T,
        //                     (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output),
        //                             self.weights2.T)
        //                      *
        //                      sigmoid_derivative(self.layer1)))
        let y_diff = y - self.last_output;
        let last_output_deriv:f32 = ACTIVATION_PRIME(self.last_output);
        // layer1.t: 4 x 1, and d_weight2: 4 x 1
        let d_weights2: Array2<f32> = self.layer1.t().mapv(|i| i * (2.0 * y_diff * last_output_deriv));

        // weight2.t(): 1 x 4, so k: 1 x 4
        let k:Array2<f32> = self.weight2.t().mapv(|i| i * 2.0 * y_diff * last_output_deriv);
        // layer1: 1 x 4, sigmoid_deriv_layer1: 1 x 4
        let layer1_deriv: Array2<f32> = self.layer1.mapv(ACTIVATION_PRIME);
        // k: 1 x 4 and sigmoid_deriv_layer1: 1 x 4, element-wise multiplication. m: 1 x 4
        let m: Array2<f32> = k * layer1_deriv;

        // "From" is only for Vec https://docs.rs/ndarray/*/ndarray/type.Array2.html
        //   let t_matrix: Array2<f32> = Array2::From(t);
        // associated item not found in `ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 2]>>`
        // Error: expected struct `ndarray::ArrayBase`, found f32

        // Because d_weights are for each element of weight1 (which is 3x4)
        // d_weights1 also needs to be Array2<f32>

        // self.lastInput is shape M=3. 
        //   If t is one-dimensional, then the operation is dot-product (scalar)
        //   If t is two-dimensional, then the operation is matrix-multiplication
        //     `self` is shape M=3, then `rhs` is shape M(3) × (2) and the result is
        //     shape *N*.
        //let last_input_t: Array2<f32> = self.lastInput.reversed_axes(); -> Not working.

        // So I had to convert all Array1 to Array2, so that we can get 2D matrix multiply
        // lastInput.t(): 3 x 1, m: 1 x 4. d_weights1: 3 x 4
        let d_weights1: Array2<f32> = self.last_input.t().dot(&m);

        // When I tried to update the weight
        //   cannot move out of borrowed content

        // In-place arithmetic operation: https://docs.rs/ndarray/0.11.2/ndarray/struct.ArrayBase.html#arithmetic-operations
        // Ensure the right-hand side is a reference.
        self.weight2 += &d_weights2;
        self.weight1 += &d_weights1;
    }
}

fn main() {
    let init_weight1 = arr2(
        &[[0.1, -0.8, 1.1, 0.3],
        [0.6, -0.2, 0.5, 1.0],
        [0.6, -0.2, 0.5, -0.4]]
    );
    let init_weight2 = arr2(&[[0.23], [0.21], [-0.5], [0.1]]);
    let mut nn = NeuralNetwork {
        last_input: arr2(&[[]]),
        weight1 : init_weight1,
        weight2 : init_weight2,
        last_output_a: 0.0,
        last_output: 0.0,
        layer1_a: arr2(&[[]]), 
        layer1 : arr2(&[[]]),
    };
    //let input1 = arr1(&[0.23, 0.21, -0.5]);
    println!("Initialized NN {:?}", nn);

    let training_data: Array2<f32> = arr2(&[
        [0., 0., 1.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.]
        ]);
    // XOR: 0., 1., 1., 0.
    let answers = vec![0., 0., 0., 1.];
//    let answers = vec![1., 1., 0.];
    //println!("training data {:?} {:?}", training_data, ans);
    
    let t_shape = training_data.shape();

    for j in 0..10000 {
        for i in 0..t_shape[0] {
            let tdata:ArrayView1<f32> = training_data.slice(s![i, ..]);
            let tmp_ans = nn.feed_forward(tdata);
            let a = answers[i];
            //nn.back_prop(a);
            nn.back_prod_prml(a);
            //println!("training {}! Diff: {}", j, tmp_ans - a);
        }
    }

    println!("Trained Network Result:");
    for i in 0..t_shape[0] {
        let tdata:ArrayView1<f32> = training_data.slice(s![i, ..]);
        let output:f32 = nn.feed_forward(tdata);
        println!("{:?} => actual {}, expected {}", tdata, output, answers[i]);
    }

}
