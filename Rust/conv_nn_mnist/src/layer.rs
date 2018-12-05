use ndarray_rand::RandomExt;
use rand::distributions::Normal;
// use rand::Rng;
use mnist::{Grayscale, Label, MnistRecord, IMG_H_SIZE, IMG_W_SIZE};
use ndarray::prelude::*;
use ndarray::IntoDimension;
use ndarray::Ix;
use ndarray::Zip;
use num_traits::identities::Zero;
use std::cmp::min;
use std::f32;
use std::f64;
use std::fmt::Debug;

lazy_static! {
    static ref INPUT_ARRAY4_ZERO: Matrix = Array::zeros((1, 1, 1, 1));
    static ref INPUT_ARRAY2_ZERO: Array2<Elem> = Array::zeros((1, 1));
}

pub fn normal_distribution(mean: f64, stddev: f64) -> Normal {
    Normal::new(mean, stddev)
}

// The implementaiton of the book leverages the layer-based implementation
// In Python, the simple RELU in the book https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
// may just be enough as Python is not statically typed language. However
// For Rust, unless you know the exact types in the function argument
// and output (input.shape), you cannot call type-specific method or operation
// Read book and identify the type!

// Looking at class Convolution, it seems input type is 4 dimensional
// https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py#L215
// FN, C, FH, FW = self.W.shape.
//   FN: number of filter. This matches the number of output map
//   C:  channel size. This matches the number of input channel size
//   FH, FW: filter height/width
//
// The Confolution uses im2col within its forward() function
// So the im2col is just helper to convert 4-dimensional tensor to 2-dimensional matrix
// so that it helps matrix multiplication easilier.
// after the multiplication and adding bias, it reshapes the output:
//        col = im2col(x, FH, FW, self.stride, self.pad)
//        col_W = self.W.reshape(FN, -1).T
//        out = np.dot(col, col_W) + self.b
//        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

pub type Elem = f64;
pub type Loss = f64;

// Following the book's interface of having 4-dimensional array as input of each layer
pub type Matrix = Array4<Elem>;
// pub type MatrixView = ArrayView4<Elem>;

// Common method that's applicable for all layer
pub trait Layer<'a> {
    fn forward(&mut self, x: &'a Matrix) -> Matrix;
    fn backward(&mut self, dout: &'a Matrix) -> Matrix;
}

pub fn mnist_to_nchw(mnist_record: &MnistRecord) -> Matrix {
    let mut ret: Array4<Elem> = Array4::<Elem>::zeros((1, 1, IMG_H_SIZE, IMG_W_SIZE));
    {
        let mut assign_mut = ret.slice_mut(s![0, 0, .., ..]);
        assign_mut.assign(&mnist_record.dots_array.mapv(f64::from));
    }
    ret
}

#[derive(Debug)]
pub struct Affine {
    original_shape: [usize; 4],
    pub weights: Array2<Elem>, // What's the dimension?
    pub d_weights: Array2<Elem>,
    last_input_matrix: Array2<Elem>, // x in the book. Owned
    pub bias: Array1<Elem>,
    pub d_bias: Array1<Elem>,
}

fn conv4d_to_2d(x: ArrayView4<Elem>) -> Array2<Elem> {
    let (n_input, _channel_size, _input_height, _input_width) = x.dim();
    // (N, channel_size*height*width)
    reshape(x, (n_input, 0))
}

impl<'a> Affine {
    pub fn new_with(initial_weights: Array2<Elem>, initial_bias: Array1<Elem>) -> Affine {
        let hidden_size = initial_weights.shape()[1];
        assert_eq!(
            hidden_size,
            initial_bias.shape()[0],
            "The 2nd part of weights and bias size should match"
        );
        Affine {
            original_shape: [0, 0, 0, 0],
            // In Numpy, the weights shape is (pool_output_size, pool_output_size) and
            //           the bias shape is (hidden_size)
            //         self.params['W2'] = weight_init_std * \
            //                 np.random.randn(pool_output_size, hidden_size)
            //         self.params['b2'] = np.zeros(hidden_size)

            // input_size for Affine is channel_size*input_height*input_width
            weights: initial_weights,
            d_weights: Array2::zeros((1, 1)),
            last_input_matrix: Array2::zeros((1, 1)),
            // The filter_num matches the number of channels in output feature map
            bias: initial_bias, // Array1::zeros(hidden_size),
            d_bias: Array1::zeros(hidden_size),
        }
    }

    pub fn new(input_size: usize, hidden_size: usize) -> Affine {
        let initial_weights = Array::random((input_size, hidden_size), normal_distribution(0., 1.));
        Self::new_with(initial_weights, Array1::zeros(hidden_size))
    }

    pub fn forward(&mut self, x: &'a Matrix) -> Array2<Elem> {
        self.forward_view(x.view())
    }
    pub fn forward_view(&mut self, x: ArrayView4<Elem>) -> Array2<Elem> {
        self.original_shape.clone_from_slice(x.shape());
        // Isn't output a matrix? It should output as (N, C, H, W) in order to feed the output
        // back to Convolution etc.
        // As per https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch08/deep_convnet.py,
        // The output of Affine never goes to Convolution layer that expects (N, C, H, W)
        self.forward_2d(&conv4d_to_2d(x))
    }
    pub fn forward_2d(&mut self, x: &'a Array2<Elem>) -> Array2<Elem> {
        debug_assert_eq!(
            x.shape()[1],
            self.weights.shape()[0],
            "The shape should match for matrix multiplication"
        );
        // (N, channel_size*height*width)
        self.last_input_matrix = x.to_owned();
        // (N, channel_size*height*width) * (channel_size*height*width, hidden_size)
        //   => input_by_weights: (N, hidden_size)
        let input_by_weights = x.dot(&self.weights);
        // self.bias: (hidden_size)
        // output: (N, hidden_size)
        let output = input_by_weights + &self.bias;
        output
    }

    pub fn backward_2d(&mut self, dout: &'a Array2<Elem>) -> Array2<Elem> {
        // dot is only available via Array2 (Matrix)..

        // self.weights: (channel_size*height*width, hidden_size)
        // self.weights.t: (hidden_size, channel_size*height*width)
        // dout: (N, hidden_size)
        // dx_matrix: (N, channel_size*height*width)
        let dx_matrix = dout.dot(&self.weights.t());

        // self.last_input_matrix: (N, channel_size*height*width)
        // self.last_input_matrix.t: (channel_size*height*width, N)
        // dout:(N, hidden_size)
        // d_weights: (channel_size*height*width, hidden_size)
        self.d_weights = self.last_input_matrix.t().dot(dout);
        self.d_bias = dout.sum_axis(Axis(0));
        dx_matrix
    }

    pub fn backward(&mut self, dout: &'a Array2<Elem>) -> Matrix {
        let dx_matrix = self.backward_2d(dout);
        reshape(
            dx_matrix.view(),
            (
                self.original_shape[0],
                self.original_shape[1],
                self.original_shape[2],
                self.original_shape[3],
            ),
        )
    }
}

pub fn reshape<E, A, D>(input: ArrayView<A, D>, shape: E) -> Array<A, E::Dim>
where
    D: Dimension,
    E: IntoDimension + Debug,
    A: Clone + Zero,
{
    let mulsum = input.shape().iter().fold(1, |sum, val| sum * val);
    let shape_str = format!("{:?}", &shape);
    let mut shape_dimension = shape.into_dimension().clone();

    let mut zero_index: i32 = -1;
    let mut mulsum_newshape = 1;
    for (i, v) in shape_dimension.slice().iter().enumerate() {
        if *v < 1 {
            debug_assert!(
                zero_index == -1,
                "Non-positive value can be passed once for the new shape"
            );
            zero_index = i as i32;
        } else {
            mulsum_newshape *= *v;
        }
    }
    if zero_index >= 0 {
        shape_dimension[zero_index as usize] = (mulsum / mulsum_newshape) as usize;
    }

    let mut input_copy = Array::zeros(input.raw_dim());
    input_copy.assign(&input);
    let reshaped_res = input_copy.into_shape(shape_dimension.into_pattern());
    match reshaped_res {
        Err(e) => {
            panic!(
                "Failed to reshape the input (shape: {:?} mulsum: {}) into {} (mulsum: {}). Error: {:?}",
                input.shape(),
                mulsum,
                shape_str,
                mulsum_newshape,
                e
            );
        }
        Ok(reshaped) => reshaped,
    }
}

#[derive(Debug)]
pub struct Pooling {
    pool_h: usize,
    pool_w: usize,
    stride: usize,
    pad: usize,
    last_input: Matrix, // x in the book. Owned?
    argmax: Array1<usize>,
}

impl<'a> Pooling {
    pub fn new(pool_h: usize, pool_w: usize, stride: usize, pad: usize) -> Pooling {
        let pooling = Pooling {
            pool_h,
            pool_w,
            stride,
            pad,
            last_input: Array4::zeros((1, 1, 1, 1)),
            argmax: Array1::zeros(1),
        };
        pooling
    }

    pub fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        (
            1 + (input_height - self.pool_h) / self.stride,
            1 + (input_width - self.pool_w) / self.stride,
        )
    }
}

impl<'a> Layer<'a> for Pooling {
    // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py#L246
    fn forward(&mut self, x: &'a Matrix) -> Matrix {
        self.last_input = x.to_owned();
        let (n_input, channel_count, input_height, input_width) = x.dim();
        let (out_h, out_w) = self.output_size(input_height, input_width);
        //let out_h = 1 + (input_height - self.pool_h) / self.stride;
        //let out_w = 1 + (input_width - self.pool_w) / self.stride;
        //        println!("input_height: {}, self.pool_h: {}, self.stride: {}", input_height, self.pool_h, self.stride);
        //        println!("input_width: {}, self.pool_w: {}, self.stride: {}", input_width, self.pool_w, self.stride);
        //        println!("x.shape: {:?}, pool_h: {}, pool_w: {}, stride: {}, pad: {}", x.shape(),
        //             self.pool_h, self.pool_w, self.stride, self.pad);
        // 'Slice end 29 is past end of axis of length 28'
        let input_col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad);
        let reshaped_col = reshape(input_col.view(), (0, self.pool_h * self.pool_w));
        // arg_max = np.argmax(col, axis=1). The return value is 1-dimension.
        self.argmax = argmax2d(&reshaped_col, Axis(1));

        // out = np.max(col, axis=1)
        let m = max2d(reshaped_col.view(), Axis(1));

        // out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        let reshaped_max = reshape(m.view(), (n_input, out_h, out_w, channel_count));
        let transposed_m = reshaped_max.permuted_axes([0, 3, 1, 2]);

        // n_input, channel_count, out_h, out_w
        transposed_m
    }

    fn backward(&mut self, dout: &'a Matrix) -> Matrix {
        // In Numpy:
        //   dout = dout.transpose(0, 2, 3, 1)

        //        println!("argmax shape: {:?}", self.argmax.shape());
        //        println!("dout shape: {:?}", dout.shape());
        let dout_transposed = dout.view().permuted_axes([0, 2, 3, 1]);

        let pool_size = self.pool_h * self.pool_w;
        let dout_size: usize = dout_transposed.len();
        let mut dmax = Array2::<Elem>::zeros((dout_size, pool_size));
        //        println!("dmax shape: {:?}", dmax.shape());

        // In Numpy:
        //   dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        //   dmax = dmax.reshape(dout.shape + (pool_size,))
        // Given dout is 4-dimensional, it's adding 5th dimension?

        // On Nov. 20th, the assignment above has incompatible left-hand and right-hand
        // What to do?
        // The left-hand, with the slice for arg_max, has elements of self.arg_max.size
        // out of the 2-dimensional array.
        // The right-hand should have the same number of element when flattened.
        let dout_flattened = reshape(dout_transposed.view(), 0);
        /*
        debug_assert_eq!(dout_flattened.len(),
        self.argmax.len()
        ); */
        for i in 0..self.argmax.len() {
            dmax[[i, self.argmax[i]]] = dout_flattened[i];
        }
        let dout_shape = dout_transposed.shape(); // 4-dimensional
        let dmax = reshape(
            dmax.view(),
            (
                dout_shape[0],
                dout_shape[1],
                dout_shape[2],
                dout_shape[3],
                pool_size,
            ),
        );

        let dmax_shape = dmax.shape();
        // dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        let dcol = reshape(
            dmax.view(),
            (dmax_shape[0] * dmax_shape[1] * dmax_shape[2], 0),
        );

        // dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        let dx = col2im(
            &dcol,
            self.last_input.shape(),
            self.pool_h,
            self.pool_w,
            self.stride,
            self.pad,
        );
        dx
    }
}

#[derive(Debug)]
pub struct SoftmaxWithLoss {
    pub y: Array2<Elem>, // output
    t: Array1<usize>,    // answers for each input
}

fn softmax_array2(x: ArrayView2<Elem>) -> Array2<Elem> {
    // [ [0.1, 0.5, 0.8],   1st data
    //   [0.3, 0.2, 0.9] ]  2nd data
    // then softmax is to make the biggest bigger, smallers to smaller:
    // [ [0.01, 0.05, 0.94],
    //   [0.03, 0.02, 0.95] ]

    let x_t = x.t();
    let x_max = max2d(x_t.view(), Axis(0));
    // Because of this subtraction, the input is transposed.
    let x_diff_max = &x_t - &x_max;
    let x_exp = x_diff_max.mapv(|x| x.exp());
    let sum_x_exp = x_exp.sum_axis(Axis(0));
    let y = x_exp / &sum_x_exp;
    y.reversed_axes()
}

fn cross_entropy_error(y: &Array2<Elem>, answer_labels: &Array1<usize>) -> Elem {
    // The first dimension is for mini-batch
    let batch_size = y.shape()[0];

    // In ths project, answer_labels hold the answer, not one-hot-vector.
    // y is one-hot-vector.
    // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/functions.py#L46
    //   If answer is given one-hot vector, convert one-hot vector to teacher labelling
    //   let answer_labels = argmax(&t, Axis(1));
    debug_assert_eq!(answer_labels.shape(), &[batch_size]);

    // For each batch, get each value of y
    // -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    // It seems this calculates values of errors across batches
    // The first time to mix data across batches

    let mut sum: Elem = 0.;
    for i in 0..batch_size {
        // 0 - 9
        let answer_index = answer_labels[i];
        let before_log = y[[i, answer_index]] + 1e-7;
        // natural logarithm: log(e). Before Nov. 28th, it was using log2
        // and caused big error
        let y_log = (before_log).ln();
        sum += y_log;
    }
    -sum / (batch_size as Elem)
}

impl SoftmaxWithLoss {
    pub fn new() -> SoftmaxWithLoss {
        let layer = SoftmaxWithLoss {
            y: Array::zeros((1, 1)),
            t: Array::zeros(1),
        };
        layer
    }
    pub fn forward(&mut self, x: &Array2<Elem>, t: &Array1<usize>) -> Elem {
        self.forward_view(x.view(), t)
    }
    pub fn forward_view(&mut self, x: ArrayView2<Elem>, t: &Array1<usize>) -> Elem {
        debug_assert_eq!(
            x.shape()[0],
            t.shape()[0],
            "The batch size of x and target does not match"
        );
        // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py#L70

        // What's softmax for 2 dimensional array?
        // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/functions.py#L31
        self.t = t.to_owned();
        self.y = softmax_array2(x);
        cross_entropy_error(&self.y, t)
    }

    pub fn backward(&mut self, _dout: Elem) -> Array2<Elem> {
        // http://cs231n.github.io/neural-networks-case-study/#grad
        let batch_size = self.t.shape()[0];
        let mut dx = self.y.to_owned();

        // In Numpy
        // When t is one-hot vector:
        //   dx = (self.y - self.t) / batch_size
        // When t is not one-hot vector (this project):
        //   dx[np.arange(batch_size), self.t] -= 1
        //   dx /= num_examples
        for i in 0..batch_size {
            dx[[i, self.t[i]]] -= 1.;
        }
        dx / (batch_size as Elem)
    }
}

#[derive(Debug)]
pub struct Relu<X>
where
    X: Dimension,
{
    mask: Array<Elem, X>, // 0. or 1.
}
pub fn relu(x: Elem) -> Elem {
    if x < 0. {
        0.
    } else {
        x
    }
}

impl Relu<Ix4> {
    pub fn new() -> Relu<Ix4> {
        let s = Relu {
            // Initialize the mask at zero
            mask: Array4::zeros((1, 1, 1, 1)),
        };
        s
    }
}

impl Relu<Ix2> {
    pub fn new() -> Relu<Ix2> {
        let s = Relu {
            // Initialize the mask at zero
            mask: Array2::zeros((1, 1)),
        };
        s
    }
}
impl<'a> Layer<'a> for Relu<Ix4> {
    fn forward(&mut self, x: &'a Matrix) -> Matrix {
        let out: Matrix = x.mapv(relu);
        self.mask = x.mapv(|x| if x < 0. { 0. } else { 1. });
        out
    }
    fn backward(&mut self, dout: &'a Matrix) -> Matrix {
        // Element-wise multiplication; 0 if mask is zero. 1 if mask is 1
        // This is not returning 1 for positive input. Is it ok?
        // Hm. Derivative was only for changing weights.
        let dx = dout * &self.mask;
        dx
    }
}
impl Relu<Ix2> {
    pub fn forward(&mut self, x: &Array2<Elem>) -> Array2<Elem> {
        let out = x.mapv(relu);
        self.mask = x.mapv(|x| if x < 0. { 0. } else { 1. });
        out
    }
    pub fn backward(&mut self, dout: &Array2<Elem>) -> Array2<Elem> {
        // Element-wise multiplication; 0 if mask is zero. 1 if mask is 1
        // This is not returning 1 for positive input. Is it ok?
        // Hm. Derivative was only for changing weights.
        let dx = dout * &self.mask;
        dx
    }
}

fn im2col(
    input: &Array4<Elem>, // n_input, channel_count, input_height, input_width
    filter_height: usize,
    filter_width: usize,
    stride: usize,
    pad: usize,
) -> Array2<Elem> {
    // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/util.py
    let (n_input, channel_count, input_height, input_width) = input.dim();
    let out_h = (input_height + 2 * pad - filter_height) / stride + 1;
    let out_w = (input_width + 2 * pad - filter_width) / stride + 1;
    //  img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    // np.pad example:
    // >>> np.pad(data, [(2,1), (0,0)], 'constant') # Add 2-pre padding and 1-post padding to first dimension
    // This input data is 2-dimensional. So the second argument takes 2-element list.
    // There's no padding for second dimension
    // array([[0, 0, 0],
    //    [0, 0, 0],
    //    [1, 2, 3],
    //    [4, 5, 6],
    //    [0, 0, 0]])
    let input_padded_height = input_height + 2 * pad;
    let input_padded_width = input_width + 2 * pad;
    let mut img: Array4<Elem> = Array4::<Elem>::zeros((
        n_input,
        channel_count,
        input_padded_height,
        input_padded_width,
    ));
    // Example:
    // input_width:5, padding: 2, then 0, 1, 2, 3, 4, 5, 6, 7, 8
    // pad..(input_width+pad)
    {
        // If pad==0, then this copy is not needed
        let mut img_init_slice_mut = img.slice_mut(s![
            ..,
            ..,
            pad..(input_height + pad),
            pad..(input_width + pad)
        ]);
        img_init_slice_mut.assign(input);
    }

    let mut col: Array6<Elem> = Array6::zeros((
        n_input,
        channel_count,
        filter_height,
        filter_width,
        out_h,
        out_w,
    ));
    for y in 0..filter_height {
        let y_max = min(y + stride * out_h, input_padded_height);
        for x in 0..filter_width {
            let x_max = min(x + stride * out_w, input_padded_width);
            // What is it doing? I think copying something. img is 4-dimensional.
            // What's the syntax of 'y:y_max:stride'?
            // col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            // What does that mean??? I need to understand assignment with slice in left-hand side
            // >>> a_3_4_5_12.shape
            // (3, 4, 5, 12)
            // >>> a_3_4_5_12[:,:,1:3:1, 4:9:1].shape
            // (3, 4, 2, 5)

            // a_3_4_6_6_5_12 =np.zeros((3,4,6,6,5,12))
            // >>> a_3_4_6_6_5_12.shape
            // (3, 4, 6, 6, 5, 12)
            // >>> a_3_4_6_6_5_12[:,:,1,4,:,:] = a_3_4_5_12[:,:,1:3:1, 4:9:1]
            // Traceback (most recent call last):
            // File "<stdin>", line 1, in <module>
            // ValueError: could not broadcast input array from shape (3,4,2,5) into shape (3,4,5,12)
            // >>> a_3_4_6_6_5_12[:,:,1,4,:,:] = a_3_4_5_12[:,:,0::1, 0::1]
            // >>> a_3_4_6_6_5_12[0][0][1][4][0][0]
            // 0.38985549989040513
            // >>> a_3_4_6_6_5_12[0][0][1][3][0][0]
            // 0.0

            // Slice to assign ndarray values at once.
            // https://docs.rs/ndarray/0.11.2/ndarray/struct.ArrayBase.html#slicing
            // and https://docs.rs/ndarray/0.12.0/ndarray/macro.s.html

            // Interestingly numpy's slice doesn't fail when max exceeds the limit
            let img_slice = img.slice(s![.., .., y..y_max;stride, x..x_max;stride]);
            let mut col_slice_mut = col.slice_mut(s![.., .., y, x, .., ..]);
            col_slice_mut.assign(&img_slice);
        }
    }
    let permuted_col = col.permuted_axes([0, 4, 5, 1, 2, 3]);

    // To avoid 'ShapeError/IncompatibleLayout: incompatible memory layout' at into_shape
    let mut permuted_col_copy: Array6<Elem> = Array6::zeros((
        n_input,
        out_h,
        out_w,
        channel_count,
        filter_height,
        filter_width,
    ));
    permuted_col_copy.assign(&permuted_col);
    // When reshape takes incompatible shape: ShapeError/IncompatibleShape: incompatible shapes
    // When input array is not c- or f-contiguous: ShapeError/IncompatibleLayout: incompatible memory layout

    let reshaped_col = permuted_col_copy.into_shape((
        n_input * out_h * out_w,
        channel_count * filter_height * filter_width,
    ));
    // Into shape https://docs.rs/ndarray/0.12.0/ndarray/struct.ArrayBase.html
    // col.permuted_axes([0, 4, 5, 1, 2, 3])
    reshaped_col.unwrap()
}
fn col2im(
    col: &Array2<Elem>,
    input_shape: &[Ix],
    filter_height: usize,
    filter_width: usize,
    stride: usize,
    pad: usize,
) -> Array4<Elem> {
    let (n_input, channel_count, input_height, input_width) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let out_h = (input_height + 2 * pad - filter_height) / stride + 1;
    let out_w = (input_width + 2 * pad - filter_width) / stride + 1;
    // col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    /*
    let mut col_copy = Array::zeros(col.raw_dim());
    col_copy.assign(col);
    let reshaped_col = col_copy.into_shape((
        n_input,
        out_h,
        out_w,
        channel_count,
        filter_height,
        filter_width,
    )); */
    let reshaped_col = reshape(
        col.view(),
        (
            n_input,
            out_h,
            out_w,
            channel_count,
            filter_height,
            filter_width,
        ),
    );
    let transposed_col = reshaped_col.permuted_axes([0, 3, 4, 5, 1, 2]);
    let mut img = Array4::zeros((
        n_input,
        channel_count,
        input_height + 2 * pad + stride - 1, // H + 2*pad + stride - 1
        input_width + 2 * pad + stride - 1,  // W + 2*pad + stride - 1
    ));
    for y in 0..filter_height {
        let y_max = y + stride * out_h;
        for x in 0..filter_width {
            let x_max = x + stride * out_w;
            let col_slice = transposed_col.slice(s![.., .., y, x, .., ..]);
            // img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            let mut img_slice_mut = img.slice_mut(s![.., .., y..y_max;stride, x..x_max;stride]);
            img_slice_mut += &col_slice;
        }
    }
    // If pad==0, then this copy is not needed
    let mut ret = Array4::<Elem>::zeros((n_input, channel_count, input_height, input_width));
    ret.assign(&img.slice(s![
        ..,
        ..,
        pad..(input_height + pad),
        pad..(input_width + pad)
    ]));
    ret
}

#[derive(Debug)]
pub struct Convolution {
    // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
    // The following indicates that weights are also 4-dimensional array
    //   self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
    filter_height: usize,
    filter_width: usize,
    stride: usize,
    pad: usize,
    last_input: Matrix, // x in the book. Owned? Yes.
    col: Array2<Elem>,
    col_weight: Array2<Elem>, // Column representation (im2col) of weights
    pub weights: Matrix,      // What's the dimension?
    pub d_weights: Matrix,
    pub bias: Array1<Elem>,
    pub d_bias: Array1<Elem>,
}

impl<'a> Convolution {
    pub fn new(
        _n_input: usize,
        filter_num: usize,
        filter_channel_count: usize,
        filter_height: usize,
        filter_width: usize,
        stride: usize,
        pad: usize,
    ) -> Convolution {
        // Initializing weights
        //   self.params['W1'] = weight_init_std * \
        //       np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        //   self.params['b1'] = np.zeros(filter_num)

        let col_weight_column_count = filter_channel_count * filter_height * filter_width;
        let conv = Convolution {
            filter_height,
            filter_width,
            last_input: Array4::zeros((1, 1, 1, 1)),
            stride,
            pad,
            col: Array2::zeros((1, 1)),
            col_weight: Array2::zeros((col_weight_column_count, filter_num)), // Do we know the H, W at initialization?
            // Filters: (FN, C, FH, FW)
            weights: Array::random(
                (
                    filter_num,
                    filter_channel_count,
                    filter_height,
                    filter_width,
                ),
                normal_distribution(0., 0.01),
            ),
            d_weights: Array4::zeros((1, 1, 1, 1)),
            // The filter_num matches the number of channels in output feature map
            bias: Array1::zeros(filter_num),
            d_bias: Array1::zeros(filter_num),
        };
        conv
    }

    pub fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let (_, _, filter_height, filter_width) = self.weights.dim();
        (
            1 + (input_height + 2 * self.pad - filter_height) / self.stride,
            1 + (input_width + 2 * self.pad - filter_width) / self.stride,
        )
    }
}

impl<'a> Layer<'a> for Convolution {
    fn forward(&mut self, x: &'a Matrix) -> Matrix {
        // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py#L214
        //  (something)x(filter_height*filter_width*channel) matrix
        let (n_input, channel_count, input_height, input_width) = x.dim();
        let (n_filter, filter_channel_count, filter_height, filter_width) = self.weights.dim();
        debug_assert_eq!(
            channel_count, filter_channel_count,
            "The number of channel in input and the number of channel in filter must match"
        );
        let (out_h, out_w) = self.output_size(input_height, input_width);
        //let out_h = 1 + (input_height + 2 * self.pad - filter_height) / self.stride;
        //let out_w = 1 + (input_width + 2 * self.pad - filter_width) / self.stride;
        // col:(rest of the right) x (filter_height * filter_width * channel_count)
        let col = im2col(
            &x,
            self.filter_height,
            self.filter_width,
            self.stride,
            self.pad,
        );

        let weight_reshaped = reshape(self.weights.view(), (n_filter, 0));
        debug_assert_eq!(
            weight_reshaped.shape()[0],
            self.bias.shape()[0],
            "The number of filter must match the number of elements in bias"
        );
        let col_weight = weight_reshaped.t();
        // In Numpy:
        //   col: something x reshaping_column_count
        //   col_weight: reshaping_column_count x n_filter

        // ndarray: inputs 90 × 75 and 250 × 3 are not compatible for matrix multiplication
        debug_assert_eq!(
            col.shape()[1],
            col_weight.shape()[0],
            "The matrix multiplication should work with these dimensions"
        );
        //   out = np.dot(col, col_W) + self.b
        let input_weight_multi = col.dot(&col_weight);
        // Error as of September 18th:
        //   darray: could not broadcast array from shape: [3] to: [90, 10]'
        debug_assert_eq!(input_weight_multi.shape()[1], self.bias.shape()[0],
        "The number of columns in input_weight_multi should match the number of elements in bias");
        let out = input_weight_multi + &self.bias;
        // In Numpy:
        //   out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        let out_reshaped = reshape(out.view(), (n_input, out_h, out_w, 0));
        let out_transposed = out_reshaped.permuted_axes([0, 3, 1, 2]);
        self.last_input = x.to_owned();
        self.col = col;
        self.col_weight.assign(&col_weight);
        
        out_transposed
        // Copy of out_transposed
        //Array::from_shape_vec(out_transposed.raw_dim(),
        //    out_transposed.iter().cloned().collect()).unwrap()
    }

    fn backward(&mut self, dout: &'a Matrix) -> Matrix {
        // FN, C, FH, FW = self.W.shape
        let (n_filter, filter_channel_count, filter_height, filter_width) = self.weights.dim();
        // permuted_axes was complaining the borrowed reference. But calling view is enough

        // In Numpy
        //   dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        let dout_transposed = dout.view().permuted_axes([0, 2, 3, 1]);

        let dout_reshaped = reshape(dout_transposed.view(), (0, n_filter)); // dout_reshaped_res.unwrap();
        // self.db = np.sum(dout, axis=0)
        self.d_bias = dout_reshaped.sum_axis(Axis(0));

        // self.dW = np.dot(self.col.T, dout)
        let col_t = self.col.t();
        let d_weight_tmp = col_t.dot(&dout_reshaped);
        let d_weight_tmp_permuted = d_weight_tmp.permuted_axes([1, 0]);
        // On December 5th, it's found that permuted_axis does not work with into_shape
        // See permuted_axes_and_reshape for detail. But adding this fix degraded the score...
        // How about learning rate 0.001?
        let d_weight_tmp_permuted = Array::from_shape_vec(d_weight_tmp_permuted.raw_dim(),
            d_weight_tmp_permuted.iter().cloned().collect()).unwrap();
        self.d_weights = d_weight_tmp_permuted
            .into_shape((n_filter, filter_channel_count, filter_height, filter_width))
            .unwrap();

        // In Numpy:
        //   dcol = np.dot(dout, self.col_W.T)
        //   dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        let d_col = dout_reshaped.dot(&self.col_weight.t());
        let dx = col2im(
            &d_col,
            self.last_input.shape(),
            filter_height,
            filter_width,
            self.stride,
            self.pad,
        );
        dx
    }
}

#[test]
fn permuted_axes_and_reshape() {
    // 2x3
    let input = arr2(&[[1, 2, 3], [4, 5, 6]]);
    // println!("input[0, 2] = {}", input[[0, 2]]); // 3

    // 3x2
    let input_permuted_axis = input.permuted_axes([1, 0]);

    // Permuted_axis has limitation: https://docs.rs/ndarray/0.12.1/ndarray/struct.ArrayBase.html#method.permuted_axes 
    // https://docs.rs/ndarray/0.12.1/ndarray/struct.ArrayBase.html#method.to_owned
    let input_permuted_axis = Array::from_shape_vec(input_permuted_axis.raw_dim(),
    input_permuted_axis.iter().cloned().collect()).unwrap();
    // println!("input_permuted_axis[2, 1] = {}", input_permuted_axis[[2, 0]]); // 3

    // Now reshaping the array
    //   input       input_permuted_axis    input_reshaped
    // [0, 0] = 1        [0, 0] = 1             [0, 0]
    // [0, 1] = 2        [0, 1] = 4             [0, 1]
    // [0, 2] = 3   ->   [1, 0] = 2      ->     [0, 2]
    // [1, 0] = 4        [1, 1] = 5             [1, 0]
    // [1, 1] = 5        [2, 0] = 3             [1, 1]
    // [1, 2] = 6        [2, 1] = 6             [1, 2]
    let input_reshaped = input_permuted_axis.into_shape((2, 3)).unwrap();
    // println!("input_reshaped[1, 1] = {}", input_reshaped[[1, 1]]);
    assert_eq!(input_reshaped[[1, 1]], 3); // Fails because it gets 4
}

#[test]
fn broadcast_assign_test() {
    let mut x = Array2::zeros((9, 3));
    let y = Array::random(3, normal_distribution(0., 1.));
    x.assign(&y);
    assert_eq!(x[[1, 1]], y[[1]]);
    // The below raises:
    // ndarray: could not broadcast array from shape: [9] to: [9, 3]
    // x.assign(&z);

    let mut x_3 = Array3::zeros((9, 3, 4));
    let y_2 = Array::random((3, 4), normal_distribution(0., 1.));
    // For each row, they're all same 3x4 matrix
    x_3.assign(&y_2);
    // below fails:
    // ndarray: could not broadcast array from shape: [3] to: [9, 3, 4]
    // x_3.assign(&y);

    // As long as the shape of last parts in suffix, it's broadcasted
    // E.g., (6, 7, 8, 9) assign (8, 9)
    //       (6, 7, 8, 9) assign (9)
    x_3.assign(&Array::random(4, normal_distribution(0., 1.)));
}

#[test]
fn slice_assign_test() {
    let mut x: Array3<Elem> = Array3::zeros((9, 3, 4));
    let y = Array::random((9, 4), normal_distribution(0., 1.));
    x.slice_mut(s![.., 2, ..]).assign(&y);
    assert_eq!(x[[0, 0, 0]], 0.);
    assert_eq!(x[[0, 2, 0]], y[[0, 0]]);
    // cargo test -- --nocapture  to show println!
}

#[test]
fn assign_test() {
    let mut x = Array4::zeros((9, 3, 100, 100));
    let y = Array::random((9, 3, 100, 100), normal_distribution(0., 1.));
    x.assign(&y);
    assert_eq!(x[[1, 1, 1, 1]], y[[1, 1, 1, 1]]);
    let z = Array::random((3, 100, 100), normal_distribution(0., 1.));
    x.assign(&z);
    for i in 0..9 {
        assert_eq!(x[[i, 1, 1, 1]], z[[1, 1, 1]]);
    }
    /*
    let m_100_100 = Array::random((100, 100), normal_distribution(0., 1.));
    x.assign(&z);
    for i in 0..9 {
        assert_eq!(x[[i,1,1,1]], m_100_100[[1,1]]);
    }*/
}

#[test]
fn im2col_stride2_test() {
    // n_input, channel_count, input_height, input_width
    let input = Array::random((100, 30, 28, 28), normal_distribution(0., 1.));
    let col2: Array2<Elem> = im2col(&input, 2, 2, 2, 0);
    assert_eq!(col2.shape(), [19600, 120]);
}

#[test]
fn im2col_shape_test() {
    /* The test below runs more than 1 minute
    let input1 = Array4::zeros((9, 3, 100, 100));
    let col1: Array2<Elem> = im2col(&input1, 50, 50, 1, 0);
    assert_eq!(col1.shape(), &[9 * 51 * 51, 50 * 50 * 3]);
    */

    // n_input, channel_count, input_height, input_width
    let input2 = Array::random((1, 3, 7, 7), normal_distribution(0., 1.));
    let col2: Array2<Elem> = im2col(&input2, 5, 5, 1, 0);
    assert_eq!(col2.shape(), [1 * 3 * 3, 5 * 5 * 3]);
    let input3 = Array::random((10, 3, 7, 7), normal_distribution(0., 1.));
    let col3: Array2<Elem> = im2col(&input3, 5, 5, 1, 0);
    assert_eq!(col3.shape(), [10 * 3 * 3, 5 * 5 * 3]);
}

#[test]
fn im2col_shape_pad_test() {
    let input4 = Array::random((1, 3, 7, 7), normal_distribution(0., 1.));
    let col4: Array2<Elem> = im2col(&input4, 5, 5, 1, 2); // pad:2
                                                          // 7/5 -> 3
                                                          // 11/5 -> 7 This is out_h and out_w
    assert_eq!(col4.shape(), [1 * 7 * 7, 5 * 5 * 3]);
}

#[test]
fn im2col_value_test() {
    let input = Array::random((10, 3, 7, 7), normal_distribution(0., 1.));
    let a = input[[1, 2, 3, 4]];
    let input_at_0 = input[[0, 0, 0, 0]];
    let col: Array2<Elem> = im2col(&input, 5, 5, 1, 0);
    assert_eq!(col.shape(), &[10 * 3 * 3, 5 * 5 * 3]);
    let b = col[[17, 57]];
    assert_eq!(col[[0, 0]], input_at_0);
    assert_eq!(b, a);
}

#[test]
fn col2im_shape_test() {
    let input = Array::random((10, 3, 7, 7), normal_distribution(0., 1.));
    let col: Array2<Elem> = im2col(&input, 5, 5, 1, 0);
    assert_eq!(col.shape(), &[10 * 3 * 3, 5 * 5 * 3]);
    let img_from_col = col2im(&col, &[10, 3, 7, 7], 5, 5, 1, 0);
    assert_eq!(img_from_col.shape(), &[10, 3, 7, 7]);
}
#[test]
fn col2im_shape_pad_test() {
    let input = Array::random((10, 3, 7, 7), normal_distribution(0., 1.));
    let col: Array2<Elem> = im2col(&input, 5, 5, 1, 2);
    let img_from_col = col2im(&col, &[10, 3, 7, 7], 5, 5, 1, 2);
    assert_eq!(img_from_col.shape(), &[10, 3, 7, 7]);
}

#[test]
fn convolution_forward_test() {
    let input = Array::random((10, 3, 7, 7), normal_distribution(0., 1.));
    let dout = Array::random((10, 30, 3, 3), normal_distribution(0., 1.));
    let dim_mul = input.shape().iter().fold(1, |sum, val| sum * val);
    assert_eq!(dim_mul, 10 * 3 * 7 * 7);
    let mut convolution_layer = Convolution::new(10, 30, 3, 5, 5, 1, 0);
    let m = convolution_layer.forward(&input);
    assert_eq!(
        m.shape(),
        dout.shape(), // &[10, 30, 3, 3]
        "(Number of input, Number of channel in output feature map,
     Number of output height, Number of outut width) should match the expected dout's shape"
    );
    let dx = convolution_layer.backward(&dout);
    assert_eq!(
        dx.shape(),
        input.shape(),
        "dx's shape must match the input's shape"
    );
}

#[test]
fn pooling_test() {
    let input = Array::random((10, 3, 7, 7), normal_distribution(0., 1.));
    let mut pooling_layer = Pooling::new(3, 3, 1, 0);
    let out = pooling_layer.forward(&input);
    let dout = &out / 10.;
    let dx = pooling_layer.backward(&dout);
    assert_eq!(out.shape(), &[10, 3, 5, 5]);
    assert_eq!(
        dx.shape(),
        &[10, 3, 7, 7],
        "The dx shape should be the same as input"
    );
}

pub fn argmax2d(input: &Array2<Elem>, axis: Axis) -> Array1<usize> {
    let find_maxarg = |a: ArrayView1<Elem>| -> usize {
        let mut ret = 0;
        let mut m = a[[ret]];
        for i in 1..a.len() {
            if a[[i]] > m {
                ret = i;
                m = a[[i]];
            }
        }
        ret
    };
    let out = input.map_axis(axis, find_maxarg);
    out
}

pub fn max2d(input: ArrayView2<Elem>, axis: Axis) -> Array1<Elem> {
    return input.fold_axis(axis, f32::MIN.into(), |m, i| (*m).max(*i));
}

#[test]
fn test_map_axis() {
    let mut input = arr2(&[[4., 1., 2.], [3., 4., 5.]]);
    // let out = input.map_axis(Axis(0), |a:ArrayView1<Elem>| a[[0]]);
    let out = argmax2d(&mut input, Axis(0));
    assert_eq!(out, arr1(&[0, 1, 1,]));
}

#[test]
fn test_relu_array4() {
    let mut input = Array::random((10, 3, 7, 7), normal_distribution(0., 1.));
    input[[1, 2, 3, 4]] = -5.;
    let dout = Array::random((10, 3, 7, 7), normal_distribution(0., 1.));

    let mut relu_layer = Relu::<Ix4>::new();
    let r = relu_layer.forward(&input);
    assert_eq!(r.shape(), &[10, 3, 7, 7]);
    let dx = relu_layer.backward(&dout);
    assert_eq!(dx.shape(), &[10, 3, 7, 7]);
    assert_eq!(
        dx[[1, 2, 3, 4]],
        0.,
        "Relu backward should give zero for minus input"
    );
}

#[test]
fn test_relu_array2() {
    let mut input = Array::random((10, 3), normal_distribution(0., 1.));
    input[[1, 2]] = -5.;
    let dout = Array::random((10, 3), normal_distribution(0., 1.));

    let mut relu_layer = Relu::<Ix2>::new();
    let r = relu_layer.forward(&input);
    assert_eq!(r.shape(), &[10, 3]);
    let dx = relu_layer.backward(&dout);
    assert_eq!(dx.shape(), &[10, 3]);
}

#[test]
fn test_affine() {
    let mut input = Array::random((10, 3, 7, 7), normal_distribution(0., 1.));
    input[[1, 2, 3, 4]] = -5.;
    let dout = Array::random((10, 100), normal_distribution(0., 1.));
    let affine_input_size = 3 * 7 * 7;
    let mut layer = Affine::new(affine_input_size, 100);
    let r = layer.forward(&input);
    assert_eq!(
        layer.original_shape,
        [10, 3, 7, 7],
        "Affine layer must remember original shape of input"
    );
    assert_eq!(r.shape(), &[10, 100]);
    let dx = layer.backward(&dout);
    assert_eq!(dx.shape(), &[10, 3, 7, 7]);
}

#[test]
fn test_softmax_with_loss() {
    let input = Array::random((10, 3), normal_distribution(0., 1.));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);
    let output = softmax_with_loss_layer.forward(&input, &answer_array1);

    let dx = softmax_with_loss_layer.backward(output);
    assert_eq!(dx.shape(), input.shape());

    // The result of softmax should sum to one
    let softmax_sum = softmax_with_loss_layer.y.sum_axis(Axis(1));
    for i in 0..10 {
        let mut s = 0.;
        for j in 0..3 {
            s += softmax_with_loss_layer.y[[i, j]];
        }
        assert_approx_eq!(s, 1.);
        assert_approx_eq!(softmax_sum[[i]], 1.);
    }
}

#[test]
fn test_softmax_array2() {
    let input = arr2(&[[0.2, 0.8, 0.1], [-0.5, 0.2, 0.9]]);
    let res = softmax_array2(input.view());
    assert_eq!(res.shape(), &[2, 3]);
    let sum = res.sum_axis(Axis(1));
    assert_eq!(sum.shape(), &[2]);

    // The sum of each row should be 1
    assert_approx_eq!(sum[[0]], 1.);
    // The sum of each row should be 1
    assert_approx_eq!(sum[[1]], 1.);

    for i in 0..3 {
        assert!(
            res[[0, i]] <= res[[0, 1]],
            "The index 1 was max for 1st data. Softmax should keep the maximum"
        );
        assert!(
            res[[1, i]] <= res[[1, 2]],
            "The index 2 was max for 2nd data. Softmax should keep the maximum"
        );
    }

    assert_approx_eq!(
        (0.2 as Elem).exp() / ((0.2 as Elem).exp() + (0.8 as Elem).exp() + (0.1 as Elem).exp()),
        res[[0, 0]] as Elem
    );
    assert_approx_eq!(
        (-0.5 as Elem).exp() / ((-0.5 as Elem).exp() + (0.2 as Elem).exp() + (0.9 as Elem).exp()),
        res[[1, 0]] as Elem
    );
    assert_approx_eq!(
        (0.9 as Elem).exp() / ((-0.5 as Elem).exp() + (0.2 as Elem).exp() + (0.9 as Elem).exp()),
        res[[1, 2]] as Elem
    );
}

#[test]
fn test_softmax_array2_minus() {
    let input = arr2(&[[-5., 4., -4.], [-50., -40., 40.]]);
    let res = softmax_array2(input.view());
    assert_eq!(res.shape(), &[2, 3]);
    let sum = res.sum_axis(Axis(1));
    assert_eq!(sum.shape(), &[2]);

    // The sum of each row should be 1
    assert_approx_eq!(sum[[0]], 1.);
    // The sum of each row should be 1
    assert_approx_eq!(sum[[1]], 1.);

    for i in 0..3 {
        assert!(
            res[[0, i]] <= res[[0, 1]],
            "The index 1 was max for 1st data. Softmax should keep the maximum"
        );
        assert!(
            res[[1, i]] <= res[[1, 2]],
            "The index 2 was max for 2nd data. Softmax should keep the maximum"
        );
    }
}

#[test]
fn test_max_array2() {
    // fn argmax(input: &Array2<Elem>, axis: Axis) -> Array1<usize> {
    let mut input = Array2::zeros((3, 4));
    //   0   0   0   0
    //   0 1.3 1.4 1.4
    // 1.2   0 1.5   0

    input[[2, 0]] = 1.2;
    input[[1, 1]] = 1.3;
    input[[2, 2]] = 1.5;
    input[[1, 2]] = 1.4;
    input[[1, 3]] = 1.4;
    let output = max2d(input.view(), Axis(0));
    assert_eq!(output, arr1(&[1.2, 1.3, 1.5, 1.4]));

    let mut input2 = Array2::zeros((3, 4));
    //  0 1.0    0   0
    //  0   0  1.2   0
    //  0   0    0 1.3
    input2[[0, 1]] = 1.;
    input2[[1, 2]] = 1.2;
    input2[[2, 3]] = 1.3;
    let output = max2d(input2.view(), Axis(1));
    assert_eq!(output, arr1(&[1.0, 1.2, 1.3]));
}

#[test]
fn test_argmax_array2() {
    // fn argmax(input: &Array2<Elem>, axis: Axis) -> Array1<usize> {
    let mut input = Array2::zeros((3, 4));
    input[[2, 0]] = 1.2;
    input[[1, 1]] = 1.3;
    input[[2, 2]] = 1.5;
    input[[1, 2]] = 1.4;
    input[[1, 3]] = 1.4;
    let output = argmax2d(&input, Axis(0));
    assert_eq!(output, arr1(&[2, 1, 2, 1]));

    let mut input2 = Array2::zeros((3, 4));
    input2[[0, 1]] = 1.;
    input2[[1, 2]] = 1.;
    input2[[2, 3]] = 1.;
    let output = argmax2d(&input2, Axis(1));
    assert_eq!(output, arr1(&[1, 2, 3]));
}

#[test]
fn test_cross_entropy_error_all_zero() {
    for i in 1..10 {
        let mut input = Array2::zeros((i, 10));
        let mut t = Array1::<usize>::zeros(i);
        for j in 0..i {
            // For i-th batch, the answer is i
            t[[j]] = j;
        }
        let ret = cross_entropy_error(&input, &t);
        // Because cross_entropy_error gives the average across the batches
        // it gives the same number for 1-size batch to 10-size batch.
        // 1e-7 is a magic number to avoid infinity
        assert_approx_eq!(ret, -(1e-7 as Elem).ln());
    }
}

#[test]
fn test_cross_entropy_error_random() {
    let mut input = Array::random((5, 10), normal_distribution(0., 1.));
    let mut t = Array1::<usize>::zeros(5);
    for i in 0..5 {
        // For i-th batch, the answer is i
        t[[i]] = i;
        input[[i, i]] = 1.;
    }

    let ret = cross_entropy_error(&input, &t);
    // Other part than the correct answer does not matter
    assert_approx_eq!(ret, 0.);
}

#[test]
fn test_cross_entropy_error_exact_match() {
    let mut input = Array2::zeros((5, 10));
    let mut t = Array1::<usize>::zeros(5);
    for i in 0..5 {
        // For i-th batch, the answer is i
        t[[i]] = i;
        input[[i, i]] = 1.;
    }
    let ret = cross_entropy_error(&input, &t);
    assert_approx_eq!(ret, 0.);
}

#[test]
fn test_differentiation_softmax_with_loss_input_all_zero() {
    let input = Array2::zeros((3, 10));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2]);

    let output = softmax_with_loss_layer.forward(&input, &answer_array1);
    assert_approx_eq!(output, 2.30258409, 0.001);
}

#[test]
fn test_differentiation_softmax_with_loss() {
    let mut input = Array::random((10, 3), normal_distribution(0., 1.));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    for _ in 0..1000 {
        let output = softmax_with_loss_layer.forward(&input, &answer_array1);
        let dx = softmax_with_loss_layer.backward(output);
        assert_eq!(dx.shape(), input.shape());
        input -= &dx;
    }
}

#[test]
fn test_differentiation_relu2() {
    let mut relu2_layer = Relu::<Ix2>::new();
    // mean 1. so that many of the data are positive
    let mut input = Array::random((10, 3), normal_distribution(1., 1.));
    let input_copy = input.to_owned();
    let mut answer = Array2::<Elem>::zeros((10, 3));
    answer[[0, 0]] = 1.;
    answer[[1, 1]] = 1.;
    answer[[2, 2]] = 1.;
    answer[[3, 0]] = 1.;
    answer[[4, 1]] = 1.;
    answer[[5, 2]] = 1.;
    answer[[6, 0]] = 1.;
    answer[[7, 1]] = 1.;
    answer[[8, 2]] = 1.;
    answer[[8, 0]] = 1.;

    for _ in 0..1000 {
        let mut output = relu2_layer.forward(&input);
        let dy = &answer - &output;
        let mut dx = relu2_layer.backward(&dy);
        dx *= 0.1;
        input += &dx;
    }
    for i in 0..10 {
        for j in 0..3 {
            // If the input is positive, then it should gradually reach 1.
            if input[[i, j]] > 0.00001 {
                assert_approx_eq!(input[[i, j]], 1.);
            } else if input[[i, j]] < 0. {
                assert_eq!(
                    input_copy[[i, j]],
                    input[[i, j]],
                    "Relu should not touch the negative input values."
                );
            }
        }
    }
}

#[test]
fn test_differentiation_relu4() {
    let mut relu_layer = Relu::<Ix4>::new();
    // mean 1. so that many of the data are positive
    let mut input = Array::random((1, 1, 5, 5), normal_distribution(1., 1.));
    let input_copy = input.to_owned();
    let mut answer = Array4::<Elem>::zeros((1, 1, 5, 5));
    for i in 0..5 {
        answer[[0, 0, i, i]] = 1.;
    }
    for _ in 0..1 {
        let mut output = relu_layer.forward(&input);
        let dy = &answer - &output;
        let mut dx = relu_layer.backward(&dy);
        input += &dx;
    }
    for i in 0..5 {
        for j in 0..5 {
            let v = input[[0, 0, i, j]];
            // If the input is positive, then it should gradually reach 1.
            if v > 0.00001 {
                assert_approx_eq!(v, 1.);
            } else if v < 0. {
                assert_eq!(
                    v,
                    input_copy[[0, 0, i, j]],
                    "Relu should not touch the negative input values."
                );
            } else {
                assert!(v < 0.00001);
            }
        }
    }
}

#[test]
fn test_differentiation_affine_input_gradient() {
    let n_input = 10;
    let mut input = Array::random((n_input, 1, 5, 5), normal_distribution(0., 1.));
    let affine_input_size = 1 * 5 * 5;
    let output_layer_size = 10;
    let mut layer = Affine::new(affine_input_size, output_layer_size);

    let learning_rate = 0.1;

    // Introducing softmax_with_loss to calculate the loss
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    for i in 0..1000 {
        let output = layer.forward(&input);
        assert_eq!(output.shape(), [n_input, output_layer_size]);
        let _loss = softmax_with_loss_layer.forward(&output, &answer_array1);
        let dout = softmax_with_loss_layer.backward(1.);
        let dx = layer.backward(&dout);

        // Adjust input
        input -= &(&dx * learning_rate);
    }
    let answer_from_layer = layer.forward(&input);
    let loss = softmax_with_loss_layer.forward(&answer_from_layer, &answer_array1);
    assert_approx_eq!(loss, 0.0, 0.01);
}

#[test]
fn test_differentiation_affine_weight_gradient() {
    let n_input = 10;
    let input = Array::random((n_input, 1, 5, 5), normal_distribution(0., 1.));
    let affine_input_size = 1 * 5 * 5;
    let output_layer_size = 10;
    let mut layer = Affine::new(affine_input_size, output_layer_size);

    let learning_rate = 0.1;

    // Introducing softmax_with_loss to calculate the loss
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    for _i in 0..1000 {
        let output = layer.forward(&input);
        assert_eq!(output.shape(), [n_input, output_layer_size]);
        let loss = softmax_with_loss_layer.forward(&output, &answer_array1);
        let dout = softmax_with_loss_layer.backward(1.);
        let _dx = layer.backward(&dout);

        // Adjust weights
        layer.weights -= &(&layer.d_weights * learning_rate);
        layer.bias -= &(&layer.d_bias * learning_rate);
    }
    let answer_from_layer = layer.forward(&input);
    let loss = softmax_with_loss_layer.forward(&answer_from_layer, &answer_array1);
    assert_approx_eq!(loss, 0.0, 0.05);
}

#[test]
fn test_differentiation_affine_sample() {
    let n_input = 1;
    let mut input = Array4::zeros((n_input, 1, 2, 2));
    // Fix input randomness along with the initial weights of the network
    // let mut input = Array::random((n_input, 1, 2, 2), normal_distribution(0., 1.));
    //[[[[-0.45449638, 0.5611855],
    //    [0.5321661, 0.22618192]]]]
    input[[0, 0, 0, 0]] = -0.45449638;
    input[[0, 0, 0, 1]] = 0.5611855;
    input[[0, 0, 1, 0]] = 0.5321661;
    input[[0, 0, 1, 1]] = 0.22618192;

    let affine_input_size = 1 * 2 * 2;
    let output_layer_size = 3;
    let mut layer = Affine::new(affine_input_size, output_layer_size);

    let mut answer = Array::random((n_input, output_layer_size), normal_distribution(0., 1.));
    answer[[0, 0]] = 0.;
    answer[[0, 1]] = 0.;
    answer[[0, 2]] = 1.;

    let learning_rate = 0.01;
    // What was wrong without learing rate?
    // It turned out that the feedback for weights was too big in absolute value
    // to gradually adjust weights
    for _ in 0..1000 {
        let output = layer.forward(&input);
        assert_eq!(output.shape(), [n_input, output_layer_size]);
        let dy = &answer - &output;
        let _dx = layer.backward(&dy);
        // Adjust weights
        layer.weights += &(&layer.d_weights * learning_rate);
        layer.bias += &(&layer.d_bias * learning_rate);
        // input += &(&dx * 0.01);
    }

    let actual = layer.forward(&input);
    Zip::from(&actual).and(&answer).apply(|a, b| {
        assert_approx_eq!(a, b, 0.01);
    });
}

#[test]
fn test_differentiation_softmax_sample() {
    let n_input = 3;
    let mut softmax_layer = SoftmaxWithLoss::new();

    // one-hot vector: the value for the index of correct answer is high
    let mut input = Array2::zeros((n_input, 10));
    // 3 inputs
    let answer_array1 = arr1(&[3, 0, 8]);

    // Having a bigger learning rate than 1 makes the final_output smaller.
    // This means something worng with softmax_layer
    let learning_rate = 5.;
    for _i in 0..1000 {
        let _output = softmax_layer.forward(&input, &answer_array1);
        // Somehow, the dout for the last layer is always 1.
        // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch07/simple_convnet.py#L129
        // It's because backward discards the argument.
        let dx = softmax_layer.backward(1.);
        // Is this appropriate to subtract dx from input? Yes, because
        // dx and dy are both positive.
        input -= &(dx * learning_rate);
    }
    let final_output = softmax_layer.forward(&input, &answer_array1);

    assert_approx_eq!(final_output, 0., 0.01);
}

#[test]
fn test_reshape_4d_2d() {
    let input = Array::random((10, 1, 5, 5), normal_distribution(0., 1.));
    let res_2d = reshape(input.view(), (10, 25));
    assert_eq!(res_2d.shape(), [10, 25]);
    let res_4d = reshape(res_2d.view(), (10, 1, 5, 5));
    assert_eq!(res_4d.shape(), [10, 1, 5, 5]);
    assert_eq!(input, res_4d);
}

#[test]
fn test_reshape_4d_1d() {
    let input = Array::random((10, 1, 5, 5), normal_distribution(0., 1.));
    let res_2d = reshape(input.view(), 0);
    assert_eq!(res_2d.shape(), [250]);
}

#[test]
fn test_reshape_4d_2d_with_minus() {
    let input = Array::random((10, 1, 5, 5), normal_distribution(0., 1.));
    let res_2d = reshape(input.view(), (10, 0));
    assert_eq!(res_2d.shape(), [10, 25]);
    let res_4d = reshape(res_2d.view(), (0, 1, 5, 5));
    assert_eq!(res_4d.shape(), [10, 1, 5, 5]);
    assert_eq!(input, res_4d);
}


#[test]
fn test_convolution_comparison_with_numpy() {
    // As convolution gradient check does not give me good number,
    // now comparing the input and output with Numpy implementation.
    // Controlling variables:
    //   - input (Array4; np.random.rand(100).reshape((1, 1, 10, 10)))
    //   - convolution weights (Array 4)
    //   - convolution bias is all zero
    //   - dout (Array4: (1, 10, 8, 8))
    let input1d = arr1(&[ 0.21291979,  0.9259537 ,  0.54908696,  0.34515395,  0.51612205,
        0.87761522,  0.85273617,  0.36854253,  0.77291418,  0.81465297,
        0.01729099,  0.84550382,  0.61240581,  0.92931396,  0.05160634,
        0.79635536,  0.56631513,  0.28312829,  0.41942054,  0.61204294,
        0.92307927,  0.85680867,  0.41359686,  0.02735984,  0.76785755,
        0.87663964,  0.98928537,  0.59126044,  0.59646155,  0.20693844,
        0.88837199,  0.41840175,  0.36849132,  0.05507745,  0.48550706,
        0.95357012,  0.16126561,  0.52719038,  0.65421994,  0.99921001,
        0.22972245,  0.35470884,  0.64564801,  0.75440485,  0.27309646,
        0.69960187,  0.08214373,  0.95316094,  0.2303191 ,  0.62778459,
        0.40772194,  0.69672341,  0.9748811 ,  0.70330979,  0.45603688,
        0.2094853 ,  0.49500349,  0.87701246,  0.41321218,  0.80196483,
        0.12951163,  0.3382637 ,  0.69591555,  0.95141834,  0.32033358,
        0.05756722,  0.81935703,  0.01493559,  0.19124027,  0.15542388,
        0.41763314,  0.3094977 ,  0.89476373,  0.14987068,  0.35374232,
        0.81641271,  0.67831094,  0.31094695,  0.1715583 ,  0.83833932,
        0.40526613,  0.66172765,  0.41454235,  0.25461563,  0.16770692,
        0.37363097,  0.46425891,  0.0011527 ,  0.9548868 ,  0.94163758,
        0.70009912,  0.0198266 ,  0.57116798,  0.38425256,  0.35444938,
        0.29766887,  0.50208431,  0.09626779,  0.96998401,  0.54368629]);
    let input = reshape(input1d.view(), (1, 1, 10, 10));

    let weights1d = arr1(&[ 0.00469583,  0.00454844,  0.00535376, -0.00914047,  0.00582796,
       -0.00147408,  0.01537375,  0.00512266,  0.00475404,  0.00372611,
       -0.00026526, -0.01284577, -0.00452465,  0.00790086,  0.00623386,
       -0.01685471,  0.0182592 , -0.00216642,  0.00663323,  0.02159165,
        0.01970806, -0.01519368,  0.00667496,  0.01605613, -0.00777775,
        0.01009396,  0.01592899, -0.00332315, -0.01160257, -0.00087018,
        0.00030449, -0.0124011 ,  0.0065012 ,  0.00773301,  0.01637796,
        0.00622416,  0.00232071,  0.0103128 ,  0.0070632 , -0.00412504,
       -0.00850421,  0.00869   , -0.01049983, -0.00850412,  0.00279871,
        0.00417925,  0.00261158, -0.00648684,  0.01044636, -0.00173756,
       -0.00740648, -0.01445288,  0.00402552, -0.01157253,  0.00668699,
        0.00296195, -0.00504207, -0.00378357,  0.00053502,  0.00300784,
       -0.0043991 , -0.00614179,  0.00546297,  0.00466345, -0.01363775,
        0.01724126,  0.00324862, -0.00287312, -0.01984624,  0.00290996,
       -0.00193967,  0.00114171, -0.00118287, -0.00241509,  0.00202042,
       -0.00944169,  0.00163119, -0.00469786, -0.01757622, -0.00389778,
        0.00340626,  0.00025613, -0.01427237, -0.01124726,  0.00078665,
        0.00407846,  0.00507927,  0.01053545,  0.00534856,  0.00562905]);
    let weights = reshape(weights1d.view(), (10, 1, 3, 3));
    let mut convolution_layer = Convolution::new(1, 10, 1, 3, 3, 1, 0);
    convolution_layer.weights.assign(&weights);
    let m = convolution_layer.forward(&input);

    let dout1d = arr1(&[ -3.74409458e-05,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   4.06068092e-04,   0.00000000e+00,
        -1.59480704e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,  -8.58517512e-05,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -8.28991065e-05,  -3.43746278e-05,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   4.92660506e-05,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         2.53977944e-06,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -3.40828005e-04,   0.00000000e+00,
         1.10340236e-04,   0.00000000e+00,   0.00000000e+00,
         3.75207012e-04,   0.00000000e+00,  -1.60690461e-04,
         0.00000000e+00,  -1.18745929e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   2.40961276e-04,   0.00000000e+00,
        -1.23641769e-04,   0.00000000e+00,   0.00000000e+00,
         1.55509412e-04,   3.05219971e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         1.40738251e-04,  -2.12006701e-04,   0.00000000e+00,
         2.18669450e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         4.74347300e-04,   0.00000000e+00,   2.74957491e-04,
         0.00000000e+00,   0.00000000e+00,   3.76621608e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   6.40961654e-05,
         0.00000000e+00,   0.00000000e+00,  -2.39803002e-04,
         0.00000000e+00,  -2.98181149e-05,   0.00000000e+00,
         3.50228279e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -3.07737446e-06,   0.00000000e+00,
        -2.53638043e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -1.14845762e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -3.76487282e-04,   1.64677823e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
        -1.23048489e-04,   0.00000000e+00,   0.00000000e+00,
        -7.60039960e-05,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   1.63620265e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
        -2.57198843e-04,   3.02781459e-04,   0.00000000e+00,
         1.26198318e-04,   0.00000000e+00,   3.20162835e-05,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         2.15940861e-04,   0.00000000e+00,   0.00000000e+00,
         2.20987768e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   1.39784198e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -5.74500588e-05,   0.00000000e+00,
        -4.14546473e-05,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   6.45371860e-07,  -8.03552960e-06,
         0.00000000e+00,   0.00000000e+00,   7.65615019e-05,
         9.72354256e-05,   0.00000000e+00,   0.00000000e+00,
        -4.08423302e-05,   0.00000000e+00,   3.21034342e-04,
         1.60803578e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
        -1.67909629e-05,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -2.24459487e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,  -1.10008587e-04,
         0.00000000e+00,   5.19122934e-05,   0.00000000e+00,
         0.00000000e+00,  -6.68411185e-05,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   7.90343566e-05,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   3.44373573e-04,
         0.00000000e+00,   0.00000000e+00,   5.32222311e-05,
         8.91096802e-05,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -2.14887789e-04,   0.00000000e+00,
         0.00000000e+00,  -2.95907185e-04,   0.00000000e+00,
         0.00000000e+00,   2.69438464e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   8.05029437e-05,
         0.00000000e+00,   0.00000000e+00,  -2.39898077e-04,
         0.00000000e+00,   2.25647819e-04,   1.96881569e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   3.92319309e-04,
         0.00000000e+00,   0.00000000e+00,  -2.60756238e-05,
         0.00000000e+00,  -3.89727351e-04,   1.68417444e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   1.75622983e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   2.24744678e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         4.79981425e-04,  -2.08641309e-04,   0.00000000e+00,
         0.00000000e+00,  -2.83486297e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   1.96309525e-04,   0.00000000e+00,
        -2.21900418e-05,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   3.59600642e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   3.37996180e-05,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         8.58634465e-06,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   4.54673079e-05,   0.00000000e+00,
         0.00000000e+00,   7.74088915e-05,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,  -2.41449067e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   4.75155559e-04,   0.00000000e+00,
         0.00000000e+00,  -1.00441623e-04,   3.37859443e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         3.49519584e-05,   0.00000000e+00,   9.35923116e-05,
         2.01403648e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
        -3.67890218e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   1.79342791e-04,   0.00000000e+00,
         0.00000000e+00,   1.09380496e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   5.54674775e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         5.12957164e-05,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         1.98076982e-04,   0.00000000e+00,   0.00000000e+00,
         4.66966623e-05,   0.00000000e+00,  -1.31199799e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         1.01639105e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -1.76823424e-05,   0.00000000e+00,
         0.00000000e+00,  -3.58839775e-05,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -2.08999091e-04,   0.00000000e+00,
        -1.34241259e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   1.95955779e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
        -3.43144247e-04,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -9.92372662e-05,   0.00000000e+00,
        -5.74113045e-05,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   1.53673731e-04,
         0.00000000e+00,  -8.32383501e-05,   0.00000000e+00,
        -1.51299764e-04,   0.00000000e+00,   1.72248459e-05,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,  -1.02757572e-05,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         2.33622620e-05,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   1.73400326e-04,
         0.00000000e+00,  -3.92436592e-05,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   2.46281975e-04,
         0.00000000e+00,  -1.73901978e-04,  -2.00038062e-04,
         0.00000000e+00,   0.00000000e+00,   1.50910525e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         1.35336474e-04,  -1.39791568e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,  -1.42758192e-04,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
        -6.26145674e-05,   0.00000000e+00,   0.00000000e+00,
         8.01648707e-05,   9.80082640e-05,   0.00000000e+00,
         0.00000000e+00,  -3.14295983e-04,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00]);
    let dout = reshape(dout1d.view(), ((1, 10, 8, 8)));
    convolution_layer.backward(&dout);
    // This number below is when running numpy's implementation with the same values
    assert_eq!(5.1617395758640978e-05, convolution_layer.d_weights[[5, 0, 2, 1]]);
}
