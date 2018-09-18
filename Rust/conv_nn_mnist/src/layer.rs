use std::fs::File;
use std::time::Instant;
use std::{thread, time};

use ansi_term::Colour::*;
use ansi_term::{ANSIString, ANSIStrings, Style};



use ndarray_rand::{RandomExt, F32};
use rand::distributions::Normal;
use rand::{thread_rng, Rng};
use std::string::ToString;
// use rand::Rng;
use ndarray::prelude::*;
use ndarray::Ix;
use utils::math::sigmoid;

const IMG_H_SIZE: usize = 28;
const IMG_W_SIZE: usize = 28;
//const HIDDEN_LAYER_SIZE: usize = 100;

const MNIST_DOT_MAX: f32 = 255.;

lazy_static! {
    static ref INPUT_ARRAY4_ZERO: Matrix = Array::zeros((1, 1, 1, 1));
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

type Elem = f32;
// Following the book's interface of having 4-dimensional array as input of each layer
pub type Matrix = Array4<Elem>;

// Common method that's applicable for all layer
pub trait Layer<'a> {
    fn forward(&mut self, x: &'a Matrix) -> Matrix;
    fn backward(&mut self, dout: Matrix) -> Matrix;
}

#[derive(Debug)]
pub struct Relu {
    mask: Matrix, // 0. or 1.
}
impl Relu {
    pub fn new() -> Relu {
        let s = Relu {
            // Initialize the mask at zero
            mask: Array4::zeros((1, 1, 1, 1)),
        };
        s
    }
    pub fn relu(x: Elem) -> Elem {
        if x < 0. {
            0.
        } else {
            x
        }
    }
}
impl<'a> Layer<'a> for Relu {
    fn forward(&mut self, x: &'a Matrix) -> Matrix {
        let out: Matrix = x.mapv(Relu::relu);
        self.mask = x.mapv(|x| if x < 0. { 0. } else { 1. });
        out
    }
    fn backward(&mut self, dout: Matrix) -> Matrix {
        // 0 if mask is zero. 1 if mask is 1
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
    let mut col_copy = Array::zeros(col.raw_dim());
    col_copy.assign(col);
    // col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    let reshaped_col = col_copy.into_shape((
        n_input,
        out_h,
        out_w,
        channel_count,
        filter_height,
        filter_width,
    ));
    let transposed_col = reshaped_col.unwrap().permuted_axes([0, 3, 4, 5, 1, 2]);
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
pub struct Convolution<'a> {
    // https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
    // The following indicates that weights are also 4-dimensional array
    //   self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
    filter_height: usize,
    filter_width: usize,
    stride: usize,
    pad: usize,
    last_input: &'a Matrix, // x in the book. Owned?
    col: Array2<Elem>,
    weights: Matrix, // What's the dimension?
    d_weights: Matrix,
    bias: Matrix,
}

impl<'a> Convolution<'a> {
    pub fn new(
        filter_height: usize,
        filter_width: usize,
        stride: usize,
        pad: usize,
    ) -> Convolution<'a> {
        let conv = Convolution {
            filter_height,
            filter_width,
            last_input: &INPUT_ARRAY4_ZERO,
            stride,
            pad,
            col: Array2::zeros((1, 1)),
            weights: Array::random((IMG_H_SIZE, IMG_W_SIZE, 1, 1), F32(Normal::new(0., 1.))),
            d_weights: Array4::zeros((1, 1, 1, 1)),
            bias: Array::random((IMG_H_SIZE, IMG_W_SIZE, 1, 1), F32(Normal::new(0., 1.))),
        };
        conv
    }
}

impl<'a> Layer<'a> for Convolution<'a> {
    fn forward(&mut self, x: &'a Matrix) -> Matrix {
        let out: Matrix = x.mapv(Relu::relu);
        //  (something)x(filter_height*filter_width*channel) matrix
        let shape = &x.shape();
        let weight_shape = &self.weights.shape();
        let (n_input, channel_count, input_height, input_width) =
            (shape[0], shape[1], shape[2], shape[3]);
        let (n_filter, filter_channel_count, filter_height, filter_width) = (
            weight_shape[0],
            weight_shape[1],
            weight_shape[2],
            weight_shape[3],
        );
        debug_assert_eq!(channel_count, filter_channel_count);
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

        let mut weight_copy = Array4::<Elem>::zeros(self.weights.raw_dim());
        weight_copy.assign(&self.weights);
        let reshaping_column_count = filter_channel_count * filter_height * filter_width;
        let weight_reshaped_res = weight_copy.into_shape((n_filter, reshaping_column_count));
        // Problem of 9/16        ^ cannot move out of borrowed content
        let weight_reshaped = weight_reshaped_res.unwrap();
        let col_weight = weight_reshaped.t();
        // col: something x reshaping_column_count
        // col_weight: reshaping_column_count x n_filter
        let out = col.dot(&col_weight) + &self.bias;
        // out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        let out_shape = &out.shape();
        let out_shape_multi_sum = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
        let out_reshaped_last_elem = out_shape_multi_sum / n_input / out_h / out_w;
        let mut out_reshaped =
            Array4::<Elem>::zeros((n_input, out_h, out_w, out_reshaped_last_elem));
        assert_eq!(
            out_reshaped.shape(),
            out.shape(),
            "Two shapes should match to assign"
        );
        out_reshaped.assign(&out);
        let out_transposed = out_reshaped.permuted_axes([0, 3, 1, 2]);
        self.last_input = x;
        self.col = col;
        out_transposed
    }
    fn backward(&mut self, dout: Matrix) -> Matrix {
        let last_input_shape = &self.last_input.shape();
        let im_from_col = col2im(
            &self.col,
            last_input_shape,
            self.filter_height,
            self.filter_width,
            1,
            0,
        );
        im_from_col
    }
}

#[test]
fn broadcast_assign_test() {
    let mut x = Array2::zeros((9, 3));
    let y = Array::random((3), F32(Normal::new(0., 1.)));
    x.assign(&y);
    assert_eq!(x[[1, 1]], y[[1]]);
    let z = Array::random((9), F32(Normal::new(0., 1.)));
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
    x_3.assign(&Array::random((4), F32(Normal::new(0., 1.))));
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
    let mut convolution_layer = Convolution::new(5, 5, 1, 0);
    let m = convolution_layer.forward(&input);
    // Error as of 9/17                ^^^^^ borrowed value does not live long enough

    println!("The result of forward: {:?}", m);
}
