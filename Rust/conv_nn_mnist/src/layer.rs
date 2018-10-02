use ndarray_rand::{RandomExt, F32};
use rand::distributions::Normal;
// use rand::Rng;
use ndarray::prelude::*;
use ndarray::Ix;
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
    weights: Array2<Elem>, // What's the dimension?
    d_weights: Array2<Elem>,
    last_input_matrix: Array2<Elem>, // x in the book. Not owned
    bias: Array1<Elem>,
    d_bias: Array1<Elem>,
}

impl<'a> Affine {
    pub fn new(input_size: usize, hidden_size: usize) -> Affine {
        let layer = Affine {
            original_shape: [0, 0, 0, 0],
            // In Numpy, the weights shape is (pool_output_size, pool_output_size) and
            //           the bias shape is (hidden_size)
            //         self.params['W2'] = weight_init_std * \
            //                 np.random.randn(pool_output_size, hidden_size)
            //         self.params['b2'] = np.zeros(hidden_size)
            weights: Array::random((input_size, hidden_size), F32(Normal::new(0., 1.))),
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
    pub fn forward(&mut self, x: &'a Matrix) -> Array2<Elem> {
        self.original_shape.clone_from_slice(x.shape());
        let (n_input, channel_size, input_height, input_width) = x.dim();
        let input_reshape_col_count = channel_size * input_height * input_width;
        //        let mut x_copy = Array4::zeros(x.raw_dim());
        //        x_copy.assign(x);
        let x_copy = x.to_owned();
        let reshaped_x_res = x_copy.into_shape((n_input, input_reshape_col_count));
        self.last_input_matrix = reshaped_x_res.unwrap();
        debug_assert_eq!(
            self.last_input_matrix.shape()[1],
            self.weights.shape()[0],
            "The shape should match for matrix multiplication"
        );
        // inputs 10 × 147 and 5 × 3 are not compatible for matrix multiplication
        let input_by_weights = self.last_input_matrix.dot(&self.weights);
        let output = input_by_weights + &self.bias;
        // Isn't output a matrix? It should output as (N, C, H, W).
        // As per https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch08/deep_convnet.py,
        // The output of Affine never goes to Convolution layer that expects (N, C, H, W)
        output
    }
    pub fn backward(&mut self, dout: &'a Array2<Elem>) -> Matrix {
        // dot is only available via Array2 (Matrix)..
        let dx_matrix = dout.dot(&self.weights.t());
        self.d_weights = self.last_input_matrix.t().dot(dout);
        self.d_bias = dout.sum_axis(Axis(0));
        let dx_reshaped_res = dx_matrix.into_shape((
            self.original_shape[0],
            self.original_shape[1],
            self.original_shape[2],
            self.original_shape[3],
        ));
        let dx_reshaped = dx_reshaped_res.unwrap();
        dx_reshaped
    }
}

#[derive(Debug)]
pub struct Pooling<'a> {
    pool_h: usize,
    pool_w: usize,
    stride: usize,
    pad: usize,
    last_input: &'a Matrix, // x in the book. Owned?
    argmax: Array1<usize>,
}

impl<'a> Pooling<'a> {
    pub fn new(pool_h: usize, pool_w: usize, stride: usize, pad: usize) -> Pooling<'a> {
        let pooling = Pooling {
            pool_h,
            pool_w,
            stride,
            pad,
            last_input: &INPUT_ARRAY4_ZERO,
            argmax: Array1::zeros(1),
        };
        pooling
    }
}

impl<'a> Layer<'a> for Pooling<'a> {
    fn forward(&mut self, x: &'a Matrix) -> Matrix {
        self.last_input = x;
        let (n_input, channel_count, input_height, input_width) = x.dim();
        let out_h = 1 + (input_height - self.pool_h) / self.stride;
        let out_w = 1 + (input_width - self.pool_w) / self.stride;
        let input_col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad);
        let col_rows_count =
            input_col.shape().iter().fold(1, |sum, val| sum * val) / (self.pool_h * self.pool_w);
        let reshaped_col_res = input_col.into_shape((col_rows_count, self.pool_h * self.pool_w));
        let reshaped_col = reshaped_col_res.unwrap();
        // arg_max = np.argmax(col, axis=1)
        self.argmax = argmax(&reshaped_col, Axis(1));

        // out = np.max(col, axis=1)
        let m = reshaped_col.fold_axis(Axis(1), -1000000., |m, i| if *i < *m { *m } else { *i });
        // out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        let reshaped_m_res = m.into_shape((n_input, out_h, out_w, channel_count));
        let transposed_m = reshaped_m_res.unwrap().permuted_axes([0, 3, 1, 2]);

        transposed_m
    }
    fn backward(&mut self, dout: &'a Matrix) -> Matrix {
        // In Numpy:
        //   dout = dout.transpose(0, 2, 3, 1)
        let dout_transposed = dout.view().permuted_axes([0, 2, 3, 1]);

        let pool_size = self.pool_h * self.pool_w;
        let dout_size: usize = dout.len();
        let dmax = Array2::<Elem>::zeros((dout_size, pool_size));

        // In Numpy:
        //   dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        //   dmax = dmax.reshape(dout.shape + (pool_size,))

        let dmax = Array4::zeros(self.last_input.raw_dim());
        dmax
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
    col_weight: Array2<Elem>, // Column representation (im2col) of weights
    weights: Matrix,          // What's the dimension?
    d_weights: Matrix,
    bias: Array1<Elem>,
    d_bias: Array1<Elem>,
}

impl<'a> Convolution<'a> {
    pub fn new(
        n_input: usize,
        filter_num: usize,
        filter_channel_count: usize,
        filter_height: usize,
        filter_width: usize,
        stride: usize,
        pad: usize,
    ) -> Convolution<'a> {
        // Initializing weights
        //   self.params['W1'] = weight_init_std * \
        //       np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        //   self.params['b1'] = np.zeros(filter_num)

        let col_weight_column_count = filter_channel_count * filter_height * filter_width;
        let conv = Convolution {
            filter_height,
            filter_width,
            last_input: &INPUT_ARRAY4_ZERO,
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
impl<'a> Layer<'a> for Convolution<'a> {
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

        let mut weight_copy = Array4::<Elem>::zeros(self.weights.raw_dim());
        weight_copy.assign(&self.weights);
        let reshaping_column_count = filter_channel_count * filter_height * filter_width;
        let weights_mulsum = self.weights.shape().iter().fold(1, |sum, val| sum * val);
        debug_assert_eq!(
            weights_mulsum,
            n_filter * reshaping_column_count,
            "The total multiplication of shapes should remain same after reshaping"
        );
        let weight_reshaped_res = weight_copy.into_shape((n_filter, reshaping_column_count));
        // Problem of 9/16        ^ cannot move out of borrowed content
        let weight_reshaped = weight_reshaped_res.unwrap();
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
        let out_shape_multi_sum = out.shape().iter().fold(1, |sum, val| sum * val);
        let out_reshaped_last_elem = out_shape_multi_sum / n_input / out_h / out_w;
        let out_reshaped_res = out.into_shape((n_input, out_h, out_w, out_reshaped_last_elem));
        let out_reshaped = out_reshaped_res.unwrap();
        let out_transposed = out_reshaped.permuted_axes([0, 3, 1, 2]);
        self.last_input = x;
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
        let dout_transposed_dim_mul = dout_transposed.shape().iter().fold(1, |sum, val| sum * val);
        let reshape_row_count = dout_transposed_dim_mul / n_filter;
        let mut dout_transposed_copy = Array::zeros(dout_transposed.raw_dim());
        dout_transposed_copy.assign(&dout_transposed);
        let dout_reshaped_res = dout_transposed_copy.into_shape((reshape_row_count, n_filter));

        // As of 9/21, "incompatible shapes" error
        let dout_reshaped = dout_reshaped_res.unwrap();

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
fn pooling_forward_test() {
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

fn argmax(input: &Array2<Elem>, axis: Axis) -> Array1<usize> {
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

#[test]
fn test_map_axis() {
    let mut input = arr2(&[[4., 1., 2.], [3., 4., 5.]]);
    // let out = input.map_axis(Axis(0), |a:ArrayView1<Elem>| a[[0]]);
    let out = argmax(&mut input, Axis(0));
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
