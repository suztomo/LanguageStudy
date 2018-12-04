use ndarray::prelude::*;

use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::Zip;
use ndarray::{arr2, arr3};
use ndarray_rand::RandomExt;
use std::fmt::Debug;

use layer::{
    argmax2d, mnist_to_nchw, normal_distribution, reshape, Affine, Convolution, Elem, Layer,
    Matrix, Pooling, Relu, SoftmaxWithLoss,
};

use network::{Network, SimpleConv};

// https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/gradient.py

fn numerical_gradient_array<D, F>(x: &Array<Elem, D>, mut f: F) -> Array<Elem, D>
where
    F: FnMut(ArrayView<Elem, D>) -> Elem,
    D: Dimension,
{
    // As per test_numerical_gradient_array2_different_h, 0.0001 gives good result for
    // our softmax_with_loss function
    numerical_gradient_array_h(x, f, 0.0001)
}

fn numerical_gradient_array_h<D, F>(x: &Array<Elem, D>, mut loss_f: F, h: f64) -> Array<Elem, D>
where
    F: FnMut(ArrayView<Elem, D>) -> f64,
    D: Dimension,
{
    let mut ret = Array::<Elem, D>::zeros(x.raw_dim());
    let mut x = x.to_owned();
    let x_iter = x.to_owned();
    for (p, _i) in x_iter.indexed_iter() {
        let dim = p.into_dimension();
        let original_element: Elem = x[dim.clone()];

        x[dim.clone()] = original_element + h;
        let fx_plus_h = loss_f(x.view());

        x[dim.clone()] = original_element - h;
        let fx_minus_h = loss_f(x.view());

        x[dim.clone()] = original_element;

        let d = fx_plus_h - fx_minus_h;
        let dx = d / (2. * h);
        ret[dim] = dx;
    }
    ret
}

fn numerical_gradient_weights_mut<D, F>(
    weights: &mut Array<Elem, D>,
    mut loss_f: F,
) -> Array<Elem, D>
where
    F: FnMut() -> f64,
    D: Dimension,
{
    let h = 0.0001;
    let mut ret = Array::<Elem, D>::zeros(weights.raw_dim());
    let mut weights = weights.to_owned();
    let weights_iter = weights.to_owned();
    for (p, _i) in weights_iter.indexed_iter() {
        let dim = p.into_dimension();
        let original_element: Elem = weights[dim.clone()];

        weights[dim.clone()] = original_element + h;
        let fx_plus_h = loss_f();

        weights[dim.clone()] = original_element - h;
        let fx_minus_h = loss_f();

        weights[dim.clone()] = original_element;

        let d = fx_plus_h - fx_minus_h;
        let dx = d / (2. * h);
        ret[dim] = dx;
    }
    ret
}

fn norm<D>(x: &Array<Elem, D>) -> f64
where
    D: Dimension,
{
    let mut sum: f64 = 0.;
    Zip::from(x).and(x).apply(|e1, e2| {
        sum += (e1 * e2) as f64;
    });
    sum.sqrt()
}

fn relative_error<D>(x1: &Array<Elem, D>, x2: &Array<Elem, D>) -> f64
where
    D: Dimension,
{
    let norm_diff = norm(&(x1 - x2));
    let norm_x1 = norm(x1);
    let norm_x2 = norm(x2);
    // http://cs231n.github.io/neural-networks-3/
    norm_diff / (norm_x1.max(norm_x2))
}

#[test]
fn test_numerical_gradient_array3() {
    // As this function knows the D=3, it works.
    let ary = arr3(&[
        [
            [1., 2., 3.], // -- 2 rows  \_
            [4., 5., 6.],
        ], // --         /
        [
            [7., 8., 9.], //            \_ 2 submatrices
            [10., 11., 12.],
        ],
    ]); //            /
    let f = |_x: ArrayView3<Elem>| -> Elem { 0.1 };
    let _ret = numerical_gradient_array(&ary, &f);
}

#[test]
fn test_numerical_gradient_different_h() {
    let mut softmax_with_loss_layer_numerical_gradient = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    let input = Array::random((10, 3), normal_distribution(0., 1.));

    let mut softmax_with_loss_layer_analytical_gradient = SoftmaxWithLoss::new();
    let _ = softmax_with_loss_layer_analytical_gradient.forward(&input, &answer_array1);
    let analytical_gradient = softmax_with_loss_layer_analytical_gradient.backward(1.);

    for i in -10..0 {
        let ten: Elem = 10.0;
        let h = ten.powi(i);

        let f = |x: ArrayView2<Elem>| -> Elem {
            let loss = softmax_with_loss_layer_numerical_gradient.forward_view(x, &answer_array1);
            loss
        };
        let numerical_gradient = numerical_gradient_array_h(&input, f, h);

        let mut max_diff: Elem = 0.;
        Zip::from(&numerical_gradient)
            .and(&analytical_gradient)
            .apply(|a, b| {
                let d = *a - *b;
                let abs_d = d.abs();
                max_diff = max_diff.max(abs_d);
                //            assert_approx_eq!(a, b, 0.1);
            });
        println!("h: {}, maximum diff: {}", h, max_diff);
    }
}

#[test]
fn test_numerical_gradient_array2() {
    let mut softmax_with_loss_layer_numerical_gradient = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1]);
    let input = arr2(&[
        [0.54041532, 0.80810253, 0.26378049],
        [0.40096443, 0.52327029, 0.84360887],
        [0.65346527, 0.16589569, 0.39340066],
        [0.37175547, 0.91136225, 0.06099962],
    ]);

    let mut softmax_with_loss_layer_analytical_gradient = SoftmaxWithLoss::new();
    let _loss = softmax_with_loss_layer_analytical_gradient.forward(&input, &answer_array1);
    // This shows 1.4896496958159746
    // Numpy shows 1.0325464866768228
    let analytical_gradient = softmax_with_loss_layer_analytical_gradient.backward(1.);
    let f = |x: ArrayView2<Elem>| -> Elem {
        let loss = softmax_with_loss_layer_numerical_gradient.forward_view(x, &answer_array1);
        loss
    };
    let numerical_gradient = numerical_gradient_array(&input, f);

    // 0.3047206 (relative error) > 1e-2 (0.01) usually means the gradient is probably wrong
    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    assert_approx_eq!(rel_error, 0., 0.000001);

    Zip::from(&numerical_gradient)
        .and(&analytical_gradient)
        .apply(|a, b| {
            assert_approx_eq!(*a, *b);
        });
}

#[test]
fn test_numerical_gradient_arraysum() {
    let input = Array::random((5, 5), normal_distribution(0., 1.));

    let f = |x: ArrayView2<Elem>| -> Elem {
        let s = x.shape();
        let mut loss = 0.;
        for i in 0..s[0] {
            for j in 0..s[1] {
                loss += x[[i, j]];
            }
        }
        loss
    };
    for i in -10..0 {
        let ten: Elem = 10.0;
        let h = ten.powi(i);
        let numerical_gradient = numerical_gradient_array_h(&input, f, h);
        assert_eq!(input.shape(), numerical_gradient.shape());

        let mut max_diff: Elem = 0.;
        Zip::from(&numerical_gradient).apply(|a| {
            max_diff = max_diff.max((a - 1.).abs());
        });
    }
}

#[test]
fn test_numerical_gradient_array4() {
    let mut affine_layer = Affine::new(25, 10);

    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    let input = Array::random((10, 1, 5, 5), normal_distribution(0., 1.));
    let f = |x: ArrayView4<Elem>| -> Elem {
        let x2 = affine_layer.forward_view(x);
        let loss = softmax_with_loss_layer.forward_view(x2.view(), &answer_array1);
        loss
    };

    let ret = numerical_gradient_array(&input, f);
    assert_eq!(input.shape(), ret.shape());
}

fn random_shape<E>(shape: E) -> Array<Elem, E::Dim>
where
    E: IntoDimension + Debug,
{
    Array::random(shape, normal_distribution(0., 1.))
}

#[test]
fn test_gradient_check_simple_convnet_affine1_weight() {
    let batch_size = 1;
    // This may need to adjust affine layer input size
    let mut simple_convnet = SimpleConv::new(batch_size, (10, 10), 3);

    // 10x3x4x3
    let input4d = random_shape((batch_size, 1, 10, 10));
    let answers = vec![1];

    let _ = simple_convnet.forward_path(input4d.to_owned(), answers.to_owned());
    let _ = simple_convnet.backward_path();
    let analytical_gradient = simple_convnet.affine_layer.d_weights.clone();

    let numerical_gradient = numerical_gradient_weights(
        &mut simple_convnet.affine_layer.weights.clone(),
        |weights_h: &Array2<Elem>| -> Elem {
            {
                let mut weights = &mut simple_convnet.affine_layer.weights;
                weights.assign(weights_h);
            }
            let l = simple_convnet.forward_path(input4d.to_owned(), answers.to_owned());
            println!("{:?}-> {}", weights_h.shape(), l);
            l
        },
    );

    // Numerical_gradient is always zero. What's wrong?

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    println!("relative error: {}", rel_error);
}


fn numerical_gradient_weights<D, F>(weights: &mut Array<Elem, D>, mut loss_f: F) -> Array<Elem, D>
where
    F: FnMut(&Array<Elem, D>) -> f64,
    D: Dimension,
{
    let h = 0.0001;
    let mut ret = Array::<Elem, D>::zeros(weights.raw_dim());
    let weights_iter = weights.to_owned();
    for (p, _i) in weights_iter.indexed_iter() {
        let dim = p.into_dimension();
        let original_element: Elem = weights[dim.clone()];

        weights[dim.clone()] = original_element + h;
        let fx_plus_h = loss_f(weights);

        weights[dim.clone()] = original_element - h;
        let fx_minus_h = loss_f(weights);

        weights[dim.clone()] = original_element;

        let d = fx_plus_h - fx_minus_h;
        let dx = d / (2. * h);
        // Dx is always zero; why?
        println!("\n\n\n\nDim at {:?} = {}", dim, dx);
        ret[dim] = dx;
    }
    ret
}

#[test]
fn test_gradient_check_simple_convnet() {
    let batch_size = 1;
    // This may need to adjust affine layer input size
    let mut simple_convnet = SimpleConv::new(batch_size, (10, 10), 30);

    // 10x3x4x3
    let input4d = random_shape((batch_size, 1, 10, 10));
    let answers = vec![1];

    let _ = simple_convnet.forward_path(input4d.to_owned(), answers.to_owned());
    let analytical_gradient = simple_convnet.backward_path();
    let numerical_gradient = numerical_gradient_array(&input4d, |x: ArrayView4<Elem>| -> Elem {
        let l = simple_convnet.forward_path(x.to_owned(), answers.to_owned());
        l
    });

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    println!("relative error: {}", rel_error);
}

#[test]
fn test_compare_numerical_gradient_convolution_input() {
    let padding = 0;
    let batch_size = 10;

    // 10x3x4x3
    let input4d = random_shape((batch_size, 1, 5, 5));

    let mut convolution_layer = Convolution::new(batch_size, 4, 1, 3, 3, 1, padding);

    let affine_weight = random_shape((36, 10));
    let mut affine_layer = Affine::new_with(affine_weight, Array1::zeros(10));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();

    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 2, 1, 2, 0, 2, 1, 1]);
    assert_eq!(answer_array1.shape()[0], batch_size);

    // 3 places to add layer
    let convolution_output = convolution_layer.forward(&input4d);
    let affine_output = affine_layer.forward(&convolution_output);
    let loss = softmax_with_loss_layer.forward(&affine_output, &answer_array1);
    let dx_loss = softmax_with_loss_layer.backward(1.);
    let dx_affine = affine_layer.backward(&dx_loss);
    let analytical_gradient = convolution_layer.backward(&dx_affine);

    let numerical_gradient = numerical_gradient_array(&input4d, |x: ArrayView4<Elem>| -> Elem {
        let x1 = convolution_layer.forward(&x.to_owned());
        let x2 = affine_layer.forward(&x1);
        let loss = softmax_with_loss_layer.forward(&x2, &answer_array1);
        loss
    });
    println!("finished numerical gradient");
    assert_eq!(input4d.shape(), numerical_gradient.shape());

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    // 0.811 with conv - affine - softmax_loss
    // assert_approx_eq!(rel_error, 0.);
}

#[test]
fn test_compare_numerical_gradient_affine4d_input() {
    // (3*4*3)x3
    let affine_weight1d = weight1d_108();
    let affine_weight = reshape(affine_weight1d.view(), (3 * 4 * 3, 3));
    let mut affine_layer = Affine::new_with(affine_weight, Array1::zeros(3));
    // Adding relu helps to reduce relative error
    let mut relu_layer = Relu::<Ix2>::new();
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();

    // 10x3x4x3
    let input1d = input1d_360();

    let input4d = reshape(input1d.view(), (10, 3, 4, 3));

    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 2, 1, 2, 0, 2, 1, 1]);

    let affine_output = affine_layer.forward(&input4d);
    let relu_output = relu_layer.forward(&affine_output);
    let _loss = softmax_with_loss_layer.forward(&relu_output, &answer_array1);

    let dx_softmax = softmax_with_loss_layer.backward(1.);
    let dx_relu = relu_layer.backward(&dx_softmax);
    let analytical_gradient = affine_layer.backward(&dx_relu);

    let numerical_gradient = numerical_gradient_array(&input4d, |x: ArrayView4<Elem>| -> Elem {
        let x1 = affine_layer.forward_view(x);
        let x2 = relu_layer.forward(&x1);
        let loss = softmax_with_loss_layer.forward(&x2, &answer_array1);
        loss
    });
    assert_eq!(input4d.shape(), numerical_gradient.shape());

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    // 0.00006666536121665446
    assert_approx_eq!(rel_error, 0., 0.0001);
}

#[test]
fn test_compare_numerical_gradient_affine4d_weights() {
    // (3*4*3)x3
    let affine_weight1d = weight1d_108();
    let affine_weight = reshape(affine_weight1d.view(), (3 * 4 * 3, 3));
    let mut affine_layer = Affine::new_with(affine_weight.to_owned(), Array1::zeros(3));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();

    // 10x3x4x3
    let input1d = input1d_360();

    let input4d = reshape(input1d.view(), (10, 3, 4, 3));

    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 2, 1, 2, 0, 2, 1, 1]);

    let affine_output = affine_layer.forward(&input4d);
    let _loss = softmax_with_loss_layer.forward(&affine_output, &answer_array1);
    let dout = softmax_with_loss_layer.backward(1.);
    let _ = affine_layer.backward(&dout);
    let analytical_gradient = affine_layer.d_weights;

    let numerical_gradient =
        numerical_gradient_array(&affine_weight, |weights: ArrayView2<Elem>| -> Elem {
            let mut affine_layer =
                Affine::new_with(weights.to_owned(), Array1::zeros(weights.shape()[1]));
            let x2 = affine_layer.forward(&input4d);
            let loss = softmax_with_loss_layer.forward(&x2, &answer_array1);
            loss
        });
    assert_eq!(affine_weight.shape(), numerical_gradient.shape());

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    // 0.00816917
    assert_approx_eq!(rel_error, 0., 0.01);
}

#[test]
fn test_compare_numerical_gradient_affine4d_bias() {
    // (3*4*3)x3
    let affine_weight1d = weight1d_108();
    let affine_weights = reshape(affine_weight1d.view(), (3 * 4 * 3, 3));
    let affine_bias = arr1(&[1., 3., 4.]);
    let mut affine_layer = Affine::new_with(affine_weights.to_owned(), affine_bias.to_owned());
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();

    // 10x3x4x3
    let input1d = input1d_360();
    let input4d = reshape(input1d.view(), (10, 3, 4, 3));

    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 2, 1, 2, 0, 2, 1, 1]);

    let affine_output = affine_layer.forward(&input4d);
    let _loss = softmax_with_loss_layer.forward(&affine_output, &answer_array1);
    let dout = softmax_with_loss_layer.backward(1.);
    let _ = affine_layer.backward(&dout);
    let analytical_gradient = affine_layer.d_bias;

    let numerical_gradient =
        numerical_gradient_array(&affine_bias, |bias_h: ArrayView1<Elem>| -> Elem {
            let mut affine_layer = Affine::new_with(affine_weights.to_owned(), bias_h.to_owned());
            let x2 = affine_layer.forward(&input4d);
            let loss = softmax_with_loss_layer.forward(&x2, &answer_array1);
            loss
        });
    assert_eq!(affine_bias.shape(), numerical_gradient.shape());

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    // 0.004502993. Too big?
    assert_approx_eq!(rel_error, 0., 0.01);
}

#[test]
fn test_compare_numerical_gradient_generic() {
    let batch_size = 1;
    // This may need to adjust affine layer input size
    let mut simple_convnet = SimpleConv::new(batch_size, (10, 10), 3);

    // 10x3x4x3
    let input4d = random_shape((batch_size, 1, 10, 10));
    let answers = vec![1];

    // 1st mutable borrow of simple_convnet
    let (mut name_weights_array2, name_weights_array1) = simple_convnet.weights_ref();
    let mut name_weights_array2: Vec<(String, &mut Array2<Elem>)> = name_weights_array2;
    for (name, weights_ref) in name_weights_array2.iter_mut() {
        let name: &String = name;
        let weights_ref: &mut Array2<Elem> = *weights_ref;
        println!("weights name: {}", name);
        numerical_gradient_weights_mut(weights_ref, || -> Elem {
            // 2nd mutable borrow of simple_convnet
            let loss = 0.; // simple_convnet.forward_path(input4d.to_owned(), answers.to_owned());
            loss
        });
    }
}

#[test]
fn test_compare_numerical_gradient_relu() {
    let mut relu_layer = Relu::<Ix2>::new();
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let input = arr2(&[
        [0.54041532, 0.80810253, -0.26378049],
        [0.40096443, -0.52327029, 0.84360887],
        [0.65346527, -0.16589569, -0.39340066],
        [-0.37175547, 0.91136225, 0.06099962],
    ]);
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1]);

    let relu_output = relu_layer.forward(&input);
    let loss = softmax_with_loss_layer.forward(&relu_output, &answer_array1);
    // This is same as Numpy
    // println!("loss for input: {}", loss);
    let dout = softmax_with_loss_layer.backward(1.);
    let analytical_gradient = relu_layer.backward(&dout);

    let numerical_gradient = numerical_gradient_array(&input, |x: ArrayView2<Elem>| -> Elem {
        let x2 = relu_layer.forward(&x.to_owned());
        let loss = softmax_with_loss_layer.forward_view(x2.view(), &answer_array1);
        loss
    });
    assert_eq!(input.shape(), numerical_gradient.shape());

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    assert_approx_eq!(rel_error, 0.);
}

#[test]
fn test_compare_numerical_gradient_affine_input() {
    let affine_weight = arr2(&[
        [0.48295846, -0.48263535, 1.51441368],
        [-0.20218527, -0.32934138, -1.06961015],
        [0.43125413, 1.02735327, -1.27496537],
    ]);
    let mut affine_layer = Affine::new_with(affine_weight, Array1::zeros(3));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let input = arr2(&[
        [0.54041532, 0.80810253, 0.26378049],
        [0.40096443, 0.52327029, 0.84360887],
        [0.65346527, 0.16589569, 0.39340066],
        [0.37175547, 0.91136225, 0.06099962],
    ]);
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1]);

    let affine_output = affine_layer.forward_2d(&input);
    let loss = softmax_with_loss_layer.forward(&affine_output, &answer_array1);
    // This is same as Numpy
    println!("loss for input: {}", loss);
    let dout = softmax_with_loss_layer.backward(1.);
    let analytical_gradient = affine_layer.backward_2d(&dout);

    let numerical_gradient = numerical_gradient_array(&input, |x: ArrayView2<Elem>| -> Elem {
        let x2 = affine_layer.forward_2d(&x.to_owned());
        let loss = softmax_with_loss_layer.forward_view(x2.view(), &answer_array1);
        loss
    });
    assert_eq!(input.shape(), numerical_gradient.shape());

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    assert_approx_eq!(rel_error, 0.);
}

#[test]
fn test_compare_numerical_gradient_affine_weights() {
    let affine_weight = arr2(&[
        [0.48295846, -0.48263535, 1.51441368],
        [-0.20218527, -0.32934138, -1.06961015],
        [0.43125413, 1.02735327, -1.27496537],
    ]);
    let mut affine_layer = Affine::new_with(affine_weight.to_owned(), Array1::zeros(3));
    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let input = arr2(&[
        [0.54041532, 0.80810253, 0.26378049],
        [0.40096443, 0.52327029, 0.84360887],
        [0.65346527, 0.16589569, 0.39340066],
        [0.37175547, 0.91136225, 0.06099962],
    ]);
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1]);

    let affine_output = affine_layer.forward_2d(&input);
    let _loss = softmax_with_loss_layer.forward(&affine_output, &answer_array1);
    let dout = softmax_with_loss_layer.backward(1.);
    let _ = affine_layer.backward_2d(&dout);
    let analytical_gradient = affine_layer.d_weights;

    let numerical_gradient =
        numerical_gradient_array(&affine_weight, |weights: ArrayView2<Elem>| -> Elem {
            let mut affine_layer =
                Affine::new_with(weights.to_owned(), Array1::zeros(weights.shape()[1]));
            let x2 = affine_layer.forward_2d(&input);
            let loss = softmax_with_loss_layer.forward_view(x2.view(), &answer_array1);
            loss
        });
    assert_eq!(affine_weight.shape(), numerical_gradient.shape());

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    assert_approx_eq!(rel_error, 0.);
}

#[test]
fn test_norm() {
    // Frobenius norm
    // https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.norm.html
    let input = arr2(&[[-4., -3., -2.], [-1., 0., 1.], [2., 3., 4.]]);
    assert_eq!(norm(&input), 7.745966692414834_f64);

    let input = arr2(&[
        [0.54041532, 0.80810253, 0.26378049],
        [0.40096443, 0.52327029, 0.84360887],
        [0.65346527, 0.16589569, 0.39340066],
        [0.37175547, 0.91136225, 0.06099962],
    ]);
    // Numpy shows 1.93461244497
    assert_approx_eq!(norm(&input), 1.93461244497_f64);
}

fn input1d_360() -> Array1<f64> {
    Array::from_vec(vec![
        -0.009348636775035203,
        0.261872431435286,
        -0.0414835901957502,
        -0.18753670140634188,
        0.3094100549358623,
        1.1443634883910936,
        1.5390671853547335,
        0.023482826064578185,
        -1.9682887379396232,
        -1.2158333408683304,
        0.45776243394449306,
        -1.1103424416840277,
        1.1405900618675606,
        -2.31819455460133,
        -1.064076699471644,
        0.5959530906752676,
        1.2876202891027948,
        1.3603344199629395,
        0.31793515143259465,
        0.29130819929294016,
        0.6064482451522232,
        2.1594683758655586,
        0.1891577378469689,
        0.623471594228428,
        1.8208667903779554,
        0.6221439551959407,
        0.731954701143978,
        0.2843294696149266,
        -0.3130540485530051,
        0.04569301352746763,
        1.0988888334497255,
        1.2343013278318014,
        -0.753450348409576,
        0.006851438075300836,
        -0.09951741137306461,
        0.09656456299537906,
        0.7705670742985294,
        0.8206676503577525,
        1.426774752499813,
        1.0975401337481785,
        -0.3375568196260969,
        -1.1862446749670887,
        0.6509370757188135,
        -1.4009911100871286,
        0.11792924275945527,
        -1.8086531815804132,
        -0.4514705860746498,
        -0.0050969412766646354,
        0.47970956264434184,
        1.2652981500447016,
        0.198805384625756,
        -0.18351408693885055,
        -0.6668005880608924,
        0.14595517189277782,
        -0.18732915366836672,
        0.7805068699096783,
        0.2698461484772622,
        -0.41595095471802,
        1.2334211096024672,
        0.7225740562697835,
        -2.300261066132335,
        -0.8374987344761269,
        0.029412960917801673,
        -0.014097962137225677,
        1.8499945731678733,
        -1.3577598955821013,
        0.4496832918084634,
        1.538979081964121,
        0.5936212351042665,
        1.192381082577089,
        -2.1987181891028222,
        0.2089894828142674,
        0.7346577639750602,
        0.3622269609301716,
        -0.17007258004246245,
        -0.49773533073304194,
        1.1154049446203549,
        -0.5291970261256093,
        0.5106170556738611,
        0.3849566300956617,
        -1.0163176427544869,
        -0.27881303328581974,
        -0.5760976534746297,
        -0.19286193927570017,
        0.1727648706060347,
        -0.08724572622015679,
        1.2209885061303718,
        0.6250548183930529,
        1.7729034746893577,
        1.1524217466972175,
        -0.8798711570624439,
        1.371049845271819,
        -0.9335277705243143,
        0.30561177340169915,
        1.1728283783455373,
        -0.014083992340723717,
        1.858098721017106,
        1.5398003950633636,
        0.1521269993107459,
        -0.8733051590355323,
        -0.057244492700933645,
        0.8800278708159137,
        -0.5956990017446439,
        -0.3109063097640534,
        -1.391640368312849,
        -0.38603005676291113,
        -0.03279727511355514,
        -0.5882688439993249,
        0.5469224964269488,
        0.08293988361458793,
        -0.98688455802353,
        -1.6469196220089832,
        -0.3935953080018228,
        0.7186520835995733,
        0.5693552308115726,
        -0.8457220303590132,
        -0.369114493578543,
        -0.023503689757975073,
        -0.020630943727992266,
        -0.9260779523247583,
        0.5536621345073488,
        -0.1826279653627171,
        0.4235683260269919,
        -0.5260725887232902,
        0.8913059017607536,
        -0.2238193178668485,
        1.553211818875609,
        -1.1515543219103368,
        1.492625673579956,
        0.5240864277367736,
        -0.034575264779029,
        -0.6449828994636181,
        -0.43668842185067347,
        -0.23940123855663062,
        -0.4788687733596952,
        -0.8230333729609866,
        -0.3460649997951255,
        -0.26298128934735626,
        1.2469430953075853,
        -0.23415853369434267,
        0.0654212891547096,
        -0.37032600809602495,
        -0.39078515517513807,
        -0.030767259668238307,
        -0.16542204231452406,
        0.40091732037097305,
        -0.8741104774418813,
        0.02623762442812843,
        2.0817427022514052,
        -0.49812867972927916,
        1.2518399715056845,
        3.1620719415993412,
        -0.9159659567856667,
        0.7182192333106797,
        0.3747500633628842,
        -0.3789777392949387,
        2.641061061322536,
        -0.14337167077374907,
        0.29551030294223085,
        -1.0903571180202243,
        -2.13993700619891,
        -1.0416953073689643,
        0.7521052020678107,
        0.8118228618115872,
        0.41054600921352036,
        -1.801005841006969,
        -0.4343652801296127,
        1.9783986853775253,
        -0.7734018616886453,
        -0.11281981496534636,
        -0.2552913296259576,
        0.43242778456730085,
        0.6732173907090827,
        -1.3245378578523535,
        -0.5650858304852868,
        1.1850685322188776,
        -0.8798391751556348,
        -0.17812818572347944,
        1.0645873041070548,
        0.8401219136718133,
        2.015529000891378,
        -0.5056948526423265,
        0.5769840190785999,
        -0.1263190900513162,
        -1.680481866605968,
        -0.7125386472599828,
        -1.3072051235099929,
        1.1723763496174944,
        -3.948344919120924,
        0.03434896401327612,
        -1.2377266407347085,
        -1.56916593698121,
        -0.9174981942075296,
        0.9898311528217142,
        0.46751067958741876,
        -0.4385160934512644,
        -0.003137618999697971,
        0.49459390842005246,
        0.6287141716563446,
        -0.16979496055985244,
        -0.24041912523328907,
        0.9346296712059543,
        -0.09739195022790338,
        -0.47582071812419713,
        -0.20752218213813003,
        0.05409432167646785,
        0.30383573961770666,
        -0.5628394371174791,
        1.66960737189554,
        -0.6717132132079763,
        -0.4091932703826586,
        0.015446118534909204,
        -0.4133165576662426,
        0.34909253174472743,
        0.2233765513458878,
        -1.6698902829815074,
        0.7860962502849647,
        0.9761268363370376,
        0.665500439338164,
        -0.8830251379967652,
        -0.0032368517102325835,
        -1.748717522035219,
        -0.21113950927198508,
        -0.6884324491097457,
        -0.532905290713359,
        -2.0642390270032007,
        -1.3599338692218466,
        -1.1377099077232207,
        -1.1611180078176104,
        -0.05306672890426677,
        0.35298276194607386,
        -0.017217304866366818,
        0.49895472931269685,
        -0.8508927898432611,
        0.4458204944916236,
        -1.071666065161356,
        -1.6051899359207797,
        0.26522928662517903,
        -1.2486200699064762,
        -0.9625794849013789,
        0.26231802766414,
        -2.5574212482786316,
        0.7059351781786547,
        -0.20470917747141834,
        0.59938158937147,
        -1.5871447360210726,
        0.5753851181366159,
        -0.5855785174246424,
        1.137696258076902,
        0.6263467597645779,
        0.11728816251546018,
        0.7598759509644688,
        -1.1646336290887023,
        -1.3570828325694686,
        -1.4141909584552863,
        0.5206613124509418,
        -1.12898798640427,
        -0.13331506344970304,
        -0.6695161020523885,
        1.0343471533647686,
        0.9505796345731344,
        1.7700792219975392,
        -0.9319259982721781,
        1.1130692416999797,
        -0.3530253616824744,
        0.1144054138951906,
        -0.08388533004251665,
        1.59903339867724,
        0.049152611492240816,
        -0.10849335751956482,
        0.3190437926096506,
        0.4601228531277616,
        0.4246836335492998,
        -1.7047218209812491,
        -1.239447832649998,
        0.7963015962739387,
        1.0183293017454724,
        1.0566718552937464,
        1.862754798206454,
        -0.7950885780409085,
        -1.1235144213791914,
        -1.1354579472154382,
        2.549975822095077,
        -1.6098313869886127,
        0.4124497605357406,
        -1.15746724214305,
        0.06172178149908283,
        0.7543348080648086,
        0.639708395154565,
        -0.8989608946686171,
        1.3323402283432386,
        -0.5228671330535628,
        0.8475570663453147,
        1.697377899752889,
        -0.0895662465589891,
        1.3020278930802898,
        2.161386859296031,
        -1.0968148340158987,
        0.811214606177866,
        1.2065029469281372,
        1.2228225976852516,
        -0.2818630390170777,
        2.3111121348605703,
        -0.643183574698085,
        1.4178347977050114,
        0.3100748800646251,
        1.1268447103461432,
        0.6651398343394609,
        0.7836596330394359,
        -0.5536036942103155,
        -0.8800878711710328,
        0.6239468969103994,
        -0.7346618164795387,
        -0.4056103977959942,
        1.4263691000768974,
        -1.7878932584066038,
        -0.7939723555879415,
        -0.7960399219098967,
        1.3150424512371122,
        -1.203685968543643,
        0.3063604677718724,
        0.3088652533751399,
        0.1420851374000475,
        -1.4989330477314438,
        -1.9553407933646747,
        -0.8634205164494964,
        0.6117503404449509,
        0.2954314795734616,
        0.5192824000231916,
        0.37720404691489945,
        -0.9562267268433715,
        -1.8859760702367416,
        0.5281351250113516,
        0.753720517989574,
        -0.402255407597849,
        -0.5798123515402313,
        2.0228140085515403,
        -2.2323157367931246,
        -1.6067515326401538,
        -0.3347607530953105,
        0.676131744808129,
        1.3304579657416344,
        -0.36018338990279103,
        -0.23535102630280338,
        -1.1820588156654885,
        1.1515248315198523,
        2.060655267537622,
        -0.16478070985946175,
        -0.003882566646484905,
        -0.16068307636425339,
        2.3423760533785596,
        -0.14958491482184116,
        1.0324740505610563,
        -1.1402175247737372,
        -0.7246489128371844,
        1.7922273238849697,
        0.9833925073196823,
        -0.7131596073905848,
        0.44078985920327207,
        -0.6860375138020224,
    ])
}

fn weight1d_108() -> Array1<f64> {
    arr1(&[
        -0.13899047254379013,
        -0.04484669902999548,
        1.1791242378353926,
        1.1373129253401544,
        -0.49421092004078926,
        1.963437342680638,
        0.03315691583223826,
        -0.2669480834489957,
        -0.5089712807737902,
        -0.25112228867056197,
        -0.11910232919103347,
        -0.19139711935108453,
        -0.40362906873396887,
        0.8612378285956889,
        -0.9820136493251618,
        -1.0989115558250913,
        -0.7788820958545758,
        -0.77659359734107,
        -0.6495082122277791,
        0.3343767287782776,
        -0.14222365142061363,
        -1.1910168088684714,
        -0.9987238755580539,
        0.7183142764925267,
        1.6997156241328726,
        -1.6549045935212976,
        0.5800260118904768,
        1.6107718484775069,
        -0.8604118548234098,
        -0.5898844018527126,
        0.37043280402706885,
        0.04741073150993697,
        1.6435502171313152,
        -1.3965025371267301,
        -0.30711741845463936,
        1.5141922385301614,
        0.632787852594581,
        0.2123690763655377,
        -0.683545526745689,
        -0.6450075581689615,
        0.2292011587356229,
        -1.1344560385269034,
        -1.1657216942748925,
        0.15151809586744677,
        -1.8410268125860585,
        0.7187648555429704,
        0.0581410923312437,
        0.37879536501694466,
        -0.9252993432189356,
        -1.669094998544574,
        0.6671091148268499,
        -1.2805461955777804,
        1.2727827032874064,
        0.43759940777156553,
        1.0235187426516597,
        1.0150969451511962,
        -1.104670851592963,
        1.266119412974307,
        -0.3572996624200624,
        -0.03109295059624584,
        0.4566064341584749,
        -0.5336481485491357,
        -0.0009675482937283285,
        -0.5787345297134922,
        1.0938225246307314,
        -0.15384131021351954,
        1.1756734637293578,
        1.0354868325823132,
        -0.43015244248803947,
        1.6479956416208885,
        0.14160161456475526,
        -0.38516066535797555,
        -0.8150943700184908,
        -0.5700073001403277,
        -1.5502940228202144,
        -0.06976847278972663,
        0.21475709193548762,
        0.7817065167326556,
        -0.18937346269303112,
        0.2113972360214565,
        -1.840500224298503,
        0.06992167409460756,
        0.6553899645224592,
        0.1458219261575918,
        -1.1581548352144422,
        -0.07660491313937506,
        0.20153192601379635,
        0.2604083876968991,
        0.10406486678391588,
        -0.6343190877089048,
        1.625588250491475,
        0.6170385822117495,
        0.9190680094174852,
        1.6577205652304778,
        0.04124981711728155,
        1.4116445973098508,
        1.2404821613963477,
        0.23465897624078377,
        1.6111701835545915,
        -0.5916421585081517,
        -0.4255489574843975,
        0.11537951507260767,
        -0.5215974731011331,
        1.977115479604823,
        1.138270182546514,
        1.1514669751639885,
        -0.20730389871746288,
        0.6421350563484785,
    ])
}
