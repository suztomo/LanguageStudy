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

// Let's say input is (a:f32, b:f32, c:f32). This is Array1<f32>
#[derive(Debug)]
struct NeuralNetwork{
    // <row count> x <column count>
    last_input: Array2<f32>, // Input 1 x 3
    weight1: Array2<f32>,   // 3 x 4
    layer1: Array2<f32>,    // 1 x 4
    weight2: Array2<f32>,   // 4 x 1
    last_output: f32, // scalar
}

fn sigmoid_derivative(x: f32) -> f32 {
    // https://beckernick.github.io/sigmoid-derivative-neural-network/
    let t = sigmoid(x);
    t * (1.0 - t)
}

impl NeuralNetwork {
    fn feed_forward(&mut self, input_array: ArrayView1<f32>) -> f32 {
        self.last_input = input_array.to_owned().insert_axis(Axis(0));
        // When theres' wrong number of rows and column, you'll get following runtime error:
        //   'ndarray: inputs 3 × 2 and 3 × 1 are not compatible for matrix multiplication'

        // lastInput: 1 x 3, weight1: 3 x 4, and thus dot_product: 1 x 4
        let dot_product:Array2<f32> = self.last_input.dot(&self.weight1);
        // https://docs.rs/ndarray/0.11.2/ndarray/struct.ArrayBase.html#method.mapv
        // layer1: 1 x 4
        self.layer1 = dot_product.mapv(sigmoid);

        // When the matrix size is different:
        //   'assertion failed: `(left == right)`
        //     left: `4`,
        //     right: `3`'
        // layer1: 1 x 4, weight2: 4 x 1, and thus output is 1 x 1 (Array2)
        let output_matrix:Array2<f32> = self.layer1.dot(&self.weight2);
        let output_before_sigmoid:f32 = output_matrix[[0, 0]];
        self.last_output = sigmoid(output_before_sigmoid);
        self.last_output
    }


    fn back_prop(&mut self, y: f32) {
        // d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        // d_weights1 = np.dot(self.input.T,
        //                     (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output),
        //                             self.weights2.T)
        //                      *
        //                      sigmoid_derivative(self.layer1)))
        let y_diff = y - self.last_output;
        let sigmoid_deriv_last_output:f32 = sigmoid_derivative(self.last_output);
        // layer1.t: 4 x 1, and d_weight2: 4 x 1
        let d_weights2: Array2<f32> = self.layer1.t().mapv(|i| i * (2.0 * y_diff * sigmoid_deriv_last_output));

        // weight2.t(): 1 x 4, so k: 1 x 4
        let k:Array2<f32> = self.weight2.t().mapv(|i| i * 2.0 * y_diff * sigmoid_deriv_last_output);
        // layer1: 1 x 4, sigmoid_deriv_layer1: 1 x 4
        let sigmoid_deriv_layer1: Array2<f32> = self.layer1.mapv(sigmoid_derivative);
        // k: 1 x 4 and sigmoid_deriv_layer1: 1 x 4, element-wise multiplication. m: 1 x 4
        let m: Array2<f32> = k * sigmoid_deriv_layer1;

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
        last_output: 0.0,
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
    let answers = vec![0., 1., 1., 0.];
    //println!("training data {:?} {:?}", training_data, ans);
    
    let t_shape = training_data.shape();

    for j in 0..1500 {
        for i in 0..t_shape[0] {
            let tdata:ArrayView1<f32> = training_data.slice(s![i, ..]);
            let tmp_ans = nn.feed_forward(tdata);
            let a = answers[i];
            nn.back_prop(a);
            println!("training {}! Diff: {}", j, tmp_ans - a);
        }
    }

    println!("Trained Network Result:");
    for i in 0..t_shape[0] {
        let tdata:ArrayView1<f32> = training_data.slice(s![i, ..]);
        let output:f32 = nn.feed_forward(tdata);
        println!("{:?} = {}", tdata, output);
    }

}
