use ndarray_rand::{RandomExt, F32};
use rand::distributions::Normal;
// use rand::Rng;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray::IntoDimension;
use ndarray::Ix;
use ndarray::Zip;
use num_traits::identities::Zero;
use std::f32;
use std::fmt::Debug;
use utils::math::sigmoid;

lazy_static! {
    static ref INPUT_ARRAY4_ZERO: Matrix = Array::zeros((1, 1, 1, 1));
    static ref INPUT_ARRAY2_ZERO: Array2<Elem> = Array::zeros((1, 1));
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

pub type Elem = f32;
// Following the book's interface of having 4-dimensional array as input of each layer
pub type Matrix = Array4<Elem>;

// Common method that's applicable for all layer
pub trait Layer<'a> {
    fn forward(&mut self, x: &'a Matrix) -> Matrix;
    fn backward(&mut self, dout: &'a Matrix) -> Matrix;
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

fn conv4d_to_2d(x: &Matrix) -> Array2<Elem> {
    let (n_input, _channel_size, _input_height, _input_width) = x.dim();
    // (N, channel_size*height*width)
    reshape(x.view(), (n_input, 0))
}

impl<'a> Affine {
    pub fn new(input_size: usize, hidden_size: usize) -> Affine {
        let initial_weights = Array::random((input_size, hidden_size), F32(Normal::new(0., 1.)));
        // Initial weights for debugging was arr2(&[[0.8520029, -1.1546916, 0.5509542],
        // [-0.74658644, 0.8143777, 0.7891202],
        // [1.1643051, -0.7081804, 0.4132015],
        // [-1.7494456, 0.9154338, -1.5571696]]);
        let layer = Affine {
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
            bias: Array1::zeros(hidden_size),
            d_bias: Array1::zeros(hidden_size),
        };
        layer
    }
    /* Can Affine implement Layer so that the layer has consistency in input
      and output shape? Problem is that Affine in the example code forwards
      Matrix (Array2), while Layer expects Array4 of (N, C, H, W)
    } 
    impl<'a> Layer<'a> for Affine {*/

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
        // Isn't output a matrix? It should output as (N, C, H, W) in order to feed the output
        // back to Convolution etc.
        // As per https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch08/deep_convnet.py,
        // The output of Affine never goes to Convolution layer that expects (N, C, H, W)
        output
    }

    pub fn forward(&mut self, x: &'a Matrix) -> Array2<Elem> {
        self.original_shape.clone_from_slice(x.shape());
        self.forward_2d(&conv4d_to_2d(&x))
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

#[derive(Debug)]
pub struct Pooling {
    pool_h: usize,
    pool_w: usize,
    stride: usize,
    pad: usize,
    last_input: Matrix, // x in the book. Owned?
    argmax: Array1<usize>,
}

fn reshape<E, A, D>(input: ArrayView<A, D>, shape: E) -> Array<A, E::Dim>
where
    D: Dimension,
    E: IntoDimension + Debug,
    A: Clone + Zero,
{
    let mulsum = input.shape().iter().fold(1, |sum, val| sum * val);
    let shape_str = format!("{:?}", &shape);
    let mut shape_dimension = shape.into_dimension().clone();

    let mut zeroIndex: i32 = -1;
    let mut mulsum_newshape = 1;
    for (i, v) in shape_dimension.slice().iter().enumerate() {
        if *v < 1 {
            debug_assert!(
                zeroIndex == -1,
                "Non-positive value can be passed once for the new shape"
            );
            zeroIndex = i as i32;
        } else {
            mulsum_newshape *= *v;
        }
    }
    if zeroIndex >= 0 {
        shape_dimension[zeroIndex as usize] = (mulsum / mulsum_newshape) as usize;
    }

    let mut input_copy = Array::zeros(input.raw_dim());
    input_copy.assign(&input);
    let reshaped_res = input_copy.into_shape(shape_dimension.into_pattern());
    match reshaped_res {
        Err(e) => {
            panic!(
                "Failed to reshape the input (shape: {:?}) into {}. Error: {:?}",
                input.shape(),
                shape_str,
                e
            );
        }
        Ok(reshaped) => reshaped,
    }
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
}

impl<'a> Layer<'a> for Pooling {
    // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py#L246
    fn forward(&mut self, x: &'a Matrix) -> Matrix {
        self.last_input = x.to_owned();
        let (n_input, channel_count, input_height, input_width) = x.dim();
        let out_h = 1 + (input_height - self.pool_h) / self.stride;
        let out_w = 1 + (input_width - self.pool_w) / self.stride;
        let input_col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad);
        let reshaped_col = reshape(input_col.view(), (0, self.pool_h * self.pool_w));
        // arg_max = np.argmax(col, axis=1). The return value is 1-dimension.
        self.argmax = argmax2d(&reshaped_col, Axis(1));

        // out = np.max(col, axis=1)
        let m = max2d(&reshaped_col.view(), Axis(1));

        // out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        let reshaped_max = reshape(m.view(), (n_input, out_h, out_w, channel_count));
        let transposed_m = reshaped_max.permuted_axes([0, 3, 1, 2]);

        transposed_m
    }
    fn backward(&mut self, dout: &'a Matrix) -> Matrix {
        // In Numpy:
        //   dout = dout.transpose(0, 2, 3, 1)

        println!("argmax shape: {:?}", self.argmax.shape());
        println!("dout shape: {:?}", dout.shape());
        let dout_transposed = dout.view().permuted_axes([0, 2, 3, 1]);

        let pool_size = self.pool_h * self.pool_w;
        let dout_size: usize = dout_transposed.len();
        let mut dmax = Array2::<Elem>::zeros((dout_size, pool_size));
        println!("dmax shape: {:?}", dmax.shape());
        // TODO: This is not implemented

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
        let dmax = Array4::<Elem>::zeros(self.last_input.raw_dim());

        //     let img_from_col = col2im(&col, &[10, 3, 7, 7], 5, 5, 1, 0);
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
    y: Array2<Elem>,  // output
    t: Array1<usize>, // answers for each input
    loss: Elem,
}

fn softmax_array2(x: &Array2<Elem>) -> Array2<Elem> {
    // [ [0.1, 0.5, 0.8],   1st data
    //   [0.3, 0.2, 0.9] ]  2nd data
    // then softmax is to make the biggest bigger, smallers to smaller:
    // [ [0.01, 0.05, 0.94],
    //   [0.03, 0.02, 0.95] ]

    let x_t = x.t();
    //  x = x - np.max(x, axis=0)
    //        let m = reshaped_col.fold_axis(Axis(1), -1000000., |m, i| if *i < *m { *m } else { *i });
    let x_max = max2d(&x_t.view(), Axis(0));
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

    let mut sum = 0.;
    for i in 0..batch_size {
        // 0 - 9
        let answer_index = answer_labels[i];
        sum += (y[[i, answer_index]] + 1e-7).log2();
    }
    -sum / (batch_size as f32)
}

impl SoftmaxWithLoss {
    pub fn new() -> SoftmaxWithLoss {
        let layer = SoftmaxWithLoss {
            y: Array::zeros((1, 1)),
            t: Array::zeros((1)),
            loss: 0.,
        };
        layer
    }
    pub fn forward(&mut self, x: &Array2<Elem>, t: &Array1<usize>) -> Elem {
        debug_assert_eq!(x.shape()[0], t.shape()[0]);
        // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py#L70

        // What's softmax for 2 dimensional array?
        // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/functions.py#L31
        self.t = t.to_owned();
        self.y = softmax_array2(x);
        self.loss = cross_entropy_error(&self.y, t);
        self.loss
    }

    pub fn backward(&mut self, _dout: Elem) -> Array2<Elem> {
        let batch_size = self.t.shape()[0];
        let mut dx = self.y.to_owned();

        // When t is one-hot vector,
        // dx = (self.y - self.t) / batch_size

        // dx[np.arange(batch_size), self.t] -= 1
        for i in 0..batch_size {
            dx[[i, self.t[i]]] -= 1.;
        }
        dx / (batch_size as f32)
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

    let mut img: Array4<Elem> = Array4::<Elem>::zeros((
        n_input,
        channel_count,
        input_height + 2 * pad,
        input_width + 2 * pad,
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
        let y_max = y + stride * out_h;
        for x in 0..filter_width {
            let x_max = x + stride * out_w;
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
        n_input: usize,
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
                F32(Normal::new(0., 1.)),
            ),
            d_weights: Array4::zeros((1, 1, 1, 1)),
            // The filter_num matches the number of channels in output feature map
            bias: Array1::zeros(filter_num),
            d_bias: Array1::zeros(filter_num),
        };
        conv
    }
}

/* Attempt to let programmers to use -1 for reshape e.g., input.reshape(FN, out_h, out_w, -1)
   usize doesn't allow -1

fn reshape(input: ArrayBase<Elem>, into_shape: &[i32]) {
    let mut shape = into_shape[:];
    // Get multiply-sum of elements excepts -1
    let mul_total_input = input.shape().iter().fold(1, |sum, val| sum * val);
    let mul_total_shape = out.shape().iter().fold(1, |sum, val| sum * (if val > 0 { val } else { 1 } ));
    for i in shape.iter_mut() {
        if (i < 0) {
            *i = mul_total_input / mul_total_shape;
        }
    }
    input.into_shape(shape)
}
*/
impl<'a> Layer<'a> for Convolution {
    fn forward(&mut self, x: &'a Matrix) -> Matrix {
        //  (something)x(filter_height*filter_width*channel) matrix
        let (n_input, channel_count, input_height, input_width) = x.dim();
        let (n_filter, filter_channel_count, filter_height, filter_width) = self.weights.dim();
        debug_assert_eq!(
            channel_count, filter_channel_count,
            "The number of channel in input and the number of channel in filter must match"
        );
        let out_h = 1 + (input_height + 2 * self.pad - filter_height) / self.stride;
        let out_w = 1 + (input_width + 2 * self.pad - filter_width) / self.stride;
        // col:(rest of the right) x (filter_height * filter_width * channel_count)
        let col = im2col(
            &x,
            self.filter_height,
            self.filter_width,
            self.stride,
            self.pad,
        );

        // let reshaping_column_count = filter_channel_count * filter_height * filter_width;
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
        let input_weight_multi = col.dot(&col_weight);
        // Error as of September 18th:
        //   darray: could not broadcast array from shape: [3] to: [90, 10]'
        debug_assert_eq!(input_weight_multi.shape()[1], self.bias.shape()[0],
        "The number of columns in input_weight_multi should match the number of elements in bias");
        let out = input_weight_multi + &self.bias;
        // In Numpy:
        //   out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        //        let out_shape = &out.shape();
        let out_reshaped = reshape(out.view(), (n_input, out_h, out_w, 0));
        let out_transposed = out_reshaped.permuted_axes([0, 3, 1, 2]);
        self.last_input = x.to_owned();
        self.col = col;
        self.col_weight.assign(&col_weight);
        //        self.col_weight = .into_owned();
        out_transposed
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
        // As of 9/21, it complains:  'ndarray: inputs 75 × 90 and 9 × 30 are not compatible for matrix multiplication'
        let col_t = self.col.t();
        let d_weight_tmp = col_t.dot(&dout_reshaped);
        self.d_weights = d_weight_tmp
            .permuted_axes([1, 0])
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
fn broadcast_assign_test() {
    let mut x = Array2::zeros((9, 3));
    let y = Array::random(3, F32(Normal::new(0., 1.)));
    x.assign(&y);
    assert_eq!(x[[1, 1]], y[[1]]);
    let z = Array::random(9, F32(Normal::new(0., 1.)));
    // The below raises:
    // ndarray: could not broadcast array from shape: [9] to: [9, 3]
    // x.assign(&z);

    let mut x_3 = Array3::zeros((9, 3, 4));
    let y_2 = Array::random((3, 4), F32(Normal::new(0., 1.)));
    // For each row, they're all same 3x4 matrix
    x_3.assign(&y_2);
    // below fails:
    // ndarray: could not broadcast array from shape: [3] to: [9, 3, 4]
    // x_3.assign(&y);

    // As long as the shape of last parts in suffix, it's broadcasted
    // E.g., (6, 7, 8, 9) assign (8, 9)
    //       (6, 7, 8, 9) assign (9)
    x_3.assign(&Array::random(4, F32(Normal::new(0., 1.))));
}

#[test]
fn slice_assign_test() {
    let mut x: Array3<f32> = Array3::zeros((9, 3, 4));
    let y = Array::random((9, 4), F32(Normal::new(0., 1.)));
    x.slice_mut(s![.., 2, ..]).assign(&y);
    assert_eq!(x[[0, 0, 0]], 0.);
    assert_eq!(x[[0, 2, 0]], y[[0, 0]]);
    // cargo test -- --nocapture  to show println!
    // println!("{:?}", x);
}

#[test]
fn assign_test() {
    let mut x = Array4::zeros((9, 3, 100, 100));
    let y = Array::random((9, 3, 100, 100), F32(Normal::new(0., 1.)));
    x.assign(&y);
    assert_eq!(x[[1, 1, 1, 1]], y[[1, 1, 1, 1]]);
    let z = Array::random((3, 100, 100), F32(Normal::new(0., 1.)));
    x.assign(&z);
    for i in 0..9 {
        assert_eq!(x[[i, 1, 1, 1]], z[[1, 1, 1]]);
    }
    /*
    let m_100_100 = Array::random((100, 100), F32(Normal::new(0., 1.)));
    x.assign(&z);
    for i in 0..9 {
        assert_eq!(x[[i,1,1,1]], m_100_100[[1,1]]);
    }*/
}

#[test]
fn im2col_shape_test() {
    /* The test below runs more than 1 minute
    let input1 = Array4::zeros((9, 3, 100, 100));
    let col1: Array2<Elem> = im2col(&input1, 50, 50, 1, 0);
    assert_eq!(col1.shape(), &[9 * 51 * 51, 50 * 50 * 3]);
    */

    // n_input, channel_count, input_height, input_width
    let input2 = Array::random((1, 3, 7, 7), F32(Normal::new(0., 1.)));
    let col2: Array2<Elem> = im2col(&input2, 5, 5, 1, 0);
    assert_eq!(col2.shape(), &[1 * 3 * 3, 5 * 5 * 3]);
    let input3 = Array::random((10, 3, 7, 7), F32(Normal::new(0., 1.)));
    let col3: Array2<Elem> = im2col(&input3, 5, 5, 1, 0);
    assert_eq!(col3.shape(), &[10 * 3 * 3, 5 * 5 * 3]);
}

#[test]
fn im2col_shape_pad_test() {
    let input4 = Array::random((1, 3, 7, 7), F32(Normal::new(0., 1.)));
    let col4: Array2<Elem> = im2col(&input4, 5, 5, 1, 2); // pad:2
                                                          // 7/5 -> 3
                                                          // 11/5 -> 7 This is out_h and out_w
    assert_eq!(col4.shape(), &[1 * 7 * 7, 5 * 5 * 3]);
}

#[test]
fn im2col_value_test() {
    let input = Array::random((10, 3, 7, 7), F32(Normal::new(0., 1.)));
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
    let input = Array::random((10, 3, 7, 7), F32(Normal::new(0., 1.)));
    let col: Array2<Elem> = im2col(&input, 5, 5, 1, 0);
    assert_eq!(col.shape(), &[10 * 3 * 3, 5 * 5 * 3]);
    let img_from_col = col2im(&col, &[10, 3, 7, 7], 5, 5, 1, 0);
    assert_eq!(img_from_col.shape(), &[10, 3, 7, 7]);
}
#[test]
fn col2im_shape_pad_test() {
    let input = Array::random((10, 3, 7, 7), F32(Normal::new(0., 1.)));
    let col: Array2<Elem> = im2col(&input, 5, 5, 1, 2);
    let img_from_col = col2im(&col, &[10, 3, 7, 7], 5, 5, 1, 2);
    assert_eq!(img_from_col.shape(), &[10, 3, 7, 7]);
}

#[test]
fn convolution_forward_test() {
    let input = Array::random((10, 3, 7, 7), F32(Normal::new(0., 1.)));
    let dout = Array::random((10, 30, 3, 3), F32(Normal::new(0., 1.)));
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
    let input = Array::random((10, 3, 7, 7), F32(Normal::new(0., 1.)));
    let dout = Array::random((10, 30, 3, 3), F32(Normal::new(0., 1.)));
    let mut pooling_layer = Pooling::new(3, 3, 1, 0);
    let r = pooling_layer.forward(&input);
    let dx = pooling_layer.backward(&dout);
    assert_eq!(r.shape(), &[10, 3, 5, 5]);
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

pub fn max2d(input: &ArrayView2<Elem>, axis: Axis) -> Array1<Elem> {
    return input.fold_axis(axis, f32::MIN, |m, i| (*m).max(*i));
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
    let mut input = Array::random((10, 3, 7, 7), F32(Normal::new(0., 1.)));
    input[[1, 2, 3, 4]] = -5.;
    let dout = Array::random((10, 3, 7, 7), F32(Normal::new(0., 1.)));

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
    let mut input = Array::random((10, 3), F32(Normal::new(0., 1.)));
    input[[1, 2]] = -5.;
    let dout = Array::random((10, 3), F32(Normal::new(0., 1.)));

    let mut relu_layer = Relu::<Ix2>::new();
    let r = relu_layer.forward(&input);
    assert_eq!(r.shape(), &[10, 3]);
    let dx = relu_layer.backward(&dout);
    assert_eq!(dx.shape(), &[10, 3]);
}

#[test]
fn test_affine() {
    let mut input = Array::random((10, 3, 7, 7), F32(Normal::new(0., 1.)));
    input[[1, 2, 3, 4]] = -5.;
    let dout = Array::random((10, 100), F32(Normal::new(0., 1.)));
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
    let input = Array::random((10, 3), F32(Normal::new(0., 1.)));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);
    let output = softmax_with_loss_layer.forward(&input, &answer_array1);

    let dx = softmax_with_loss_layer.backward(output);
    assert_eq!(dx.shape(), input.shape());
}

#[test]
fn test_softmax_array2() {
    let input = arr2(&[[0.2, 0.8, 0.1], [-0.5, 0.2, 0.9]]);
    let res = softmax_array2(&input);
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
        (0.2 as f32).exp() / ((0.2 as f32).exp() + (0.8 as f32).exp() + (0.1 as f32).exp()),
        res[[0, 0]] as f32
    );
    assert_approx_eq!(
        (-0.5 as f32).exp() / ((-0.5 as f32).exp() + (0.2 as f32).exp() + (0.9 as f32).exp()),
        res[[1, 0]] as f32
    );
    assert_approx_eq!(
        (0.9 as f32).exp() / ((-0.5 as f32).exp() + (0.2 as f32).exp() + (0.9 as f32).exp()),
        res[[1, 2]] as f32
    );
}

#[test]
fn test_softmax_array2_minus() {
    let input = arr2(&[[-5., 4., -4.], [-50., -40., 40.]]);
    let res = softmax_array2(&input);
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
    let output = max2d(&input.view(), Axis(0));
    assert_eq!(output, arr1(&[1.2, 1.3, 1.5, 1.4]));

    let mut input2 = Array2::zeros((3, 4));
    //  0 1.0    0   0
    //  0   0  1.2   0
    //  0   0    0 1.3
    input2[[0, 1]] = 1.;
    input2[[1, 2]] = 1.2;
    input2[[2, 3]] = 1.3;
    let output = max2d(&input2.view(), Axis(1));
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
        assert_approx_eq!(ret, -(1e-7 as f32).log2());
    }
}

#[test]
fn test_cross_entropy_error_random() {
    let mut input = Array::random((5, 10), F32(Normal::new(0., 1.)));
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
    let mut input = Array2::zeros((3, 10));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2]);

    let output = softmax_with_loss_layer.forward(&input, &answer_array1);
    assert_approx_eq!(output, 3.3219, 0.001);
}

#[test]
fn test_differentiation_softmax_with_loss() {
    let mut input = Array::random((10, 3), F32(Normal::new(0., 1.)));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    for i in 0..1000 {
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
    let mut input = Array::random((10, 3), F32(Normal::new(1., 1.)));
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

    for i in 0..1000 {
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
    let mut input = Array::random((1, 1, 5, 5), F32(Normal::new(1., 1.)));
    let input_copy = input.to_owned();
    let mut answer = Array4::<Elem>::zeros((1, 1, 5, 5));
    for i in 0..5 {
        answer[[0, 0, i, i]] = 1.;
    }
    for i in 0..1 {
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
fn test_differentiation_affine() {
    let n_input = 1;
    let mut input = Array::random((n_input, 1, 5, 5), F32(Normal::new(0., 1.)));
    let affine_input_size = 1 * 5 * 5;
    let output_layer_size = 10;
    let mut layer = Affine::new(affine_input_size, output_layer_size);

    let answer = Array::random((n_input, output_layer_size), F32(Normal::new(0., 1.)));
    let learning_rate = 0.01;

    for _i in 0..1000 {
        let output = layer.forward(&input);
        assert_eq!(output.shape(), [n_input, output_layer_size]);
        //println!("output: {:?}\n\nanswer:{:?}\n", output, answer);
        let dy = &answer - &output;
        //println!("dy: {:?}\n", dy);
        let dx = layer.backward(&dy);
        //println!("dx: {:?}", dx);
        //println!("d_weights: {:?}", layer.d_weights);

        // Adjust weights
        layer.weights += &(&layer.d_weights * learning_rate);
        layer.bias += &(&layer.d_bias * learning_rate);
        //input -= &(&dx * learning_rate);
    }
    let answer_from_layer = layer.forward(&input);
    Zip::from(&answer_from_layer).and(&answer).apply(|a, b| {
        assert_approx_eq!(a, b, 0.01);
    });
}

#[test]
fn test_differentiation_affine_sample() {
    let n_input = 1;
    let mut input = Array4::zeros((n_input, 1, 2, 2));
    // Fix input randomness along with the initial weights of the network
    // let mut input = Array::random((n_input, 1, 2, 2), F32(Normal::new(0., 1.)));
    //[[[[-0.45449638, 0.5611855],
    //    [0.5321661, 0.22618192]]]]
    input[[0, 0, 0, 0]] = -0.45449638;
    input[[0, 0, 0, 1]] = 0.5611855;
    input[[0, 0, 1, 0]] = 0.5321661;
    input[[0, 0, 1, 1]] = 0.22618192;

    let affine_input_size = 1 * 2 * 2;
    let output_layer_size = 3;
    let mut layer = Affine::new(affine_input_size, output_layer_size);

    let mut answer = Array::random((n_input, output_layer_size), F32(Normal::new(0., 1.)));
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
        let dx = layer.backward(&dy);
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
    // Fix input randomness along with the initial weights of the network
    // They're almost zero
    // let mut input = Array::random((n_input, 10), F32(Normal::new(0., 0.5)));
    //[[[[-0.45449638, 0.5611855],
    //    [0.5321661, 0.22618192]]]]
    // 3 inputs
    let answer_array1 = arr1(&[3, 0, 8]);

    let learning_rate = 1.;
    for i in 0..1 {
        let output = softmax_layer.forward(&input, &answer_array1);

        // Somehow, the dout for the last layer is always 1.
        // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch07/simple_convnet.py#L129
        let dx = softmax_layer.backward(1.);

        // Is this appropriate to subtract dx from input?
        input -= &(dx * learning_rate);
    }
    println!("input value? : {:?}", &input);
    let final_output = softmax_layer.forward(&input, &answer_array1);

    println!(
        "Input got adjusted? (smaller the better) : {:?}",
        final_output
    );
}

#[test]
fn test_reshape_4d_2d() {
    let input = Array::random((10, 1, 5, 5), F32(Normal::new(0., 1.)));
    let res_2d = reshape(input.view(), (10, 25));
    assert_eq!(res_2d.shape(), [10, 25]);
    let res_4d = reshape(res_2d.view(), (10, 1, 5, 5));
    assert_eq!(res_4d.shape(), [10, 1, 5, 5]);
    assert_eq!(input, res_4d);
}

#[test]
fn test_reshape_4d_1d() {
    let input = Array::random((10, 1, 5, 5), F32(Normal::new(0., 1.)));
    let res_2d = reshape(input.view(), 0);
    assert_eq!(res_2d.shape(), [250]);
}

#[test]
fn test_reshape_4d_2d_with_minus() {
    let input = Array::random((10, 1, 5, 5), F32(Normal::new(0., 1.)));
    let res_2d = reshape(input.view(), (10, 0));
    assert_eq!(res_2d.shape(), [10, 25]);
    let res_4d = reshape(res_2d.view(), (0, 1, 5, 5));
    assert_eq!(res_4d.shape(), [10, 1, 5, 5]);
    assert_eq!(input, res_4d);
}
