use ndarray::prelude::*;

use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::Ix;
use ndarray::Zip;
use ndarray::{arr2, arr3};
use ndarray_rand::{RandomExt};
use num_traits::identities::Zero;
use rand::distributions::Normal;
use std::cmp::max;

use layer::{argmax2d, Affine, Convolution, Elem, Layer, Matrix, Pooling, Relu, SoftmaxWithLoss, F64};

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
    for (p, i) in x_iter.indexed_iter() {
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

fn norm<D>(x: &Array<Elem, D>) -> f64
where D: Dimension {
    let mut sum:f64 = 0.;
    Zip::from(x)
    .and(x)
    .apply(|e1, e2| {
        sum += (e1 * e2) as f64;
    });
    sum.sqrt()
}

fn relative_error<D>(x1: &Array<Elem, D>, x2: &Array<Elem, D>) -> f64
where D: Dimension {
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
    let f = |x: ArrayView3<Elem>| -> Elem { 0.1 };
    let _ret = numerical_gradient_array(&ary, &f);
}

#[test]
fn test_numerical_gradient_different_h() {
    let mut softmax_with_loss_layer_numerical_gradient = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    let input = Array::random((10, 3), F64(Normal::new(0., 1.)));

    let mut softmax_with_loss_layer_analytical_gradient = SoftmaxWithLoss::new();
    let _ = softmax_with_loss_layer_analytical_gradient.forward(&input, &answer_array1);
    let analytical_gradient = softmax_with_loss_layer_analytical_gradient.backward(1.);

    for i in -10..0 {
        let ten:Elem = 10.0;
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
    let input = arr2(&[[ 0.54041532, 0.80810253, 0.26378049],
        [ 0.40096443, 0.52327029, 0.84360887],
        [ 0.65346527, 0.16589569, 0.39340066],
        [ 0.37175547, 0.91136225, 0.06099962]]);

    let mut softmax_with_loss_layer_analytical_gradient = SoftmaxWithLoss::new();
    let loss = softmax_with_loss_layer_analytical_gradient.forward(&input, &answer_array1);
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
    let input = Array::random((5, 5), F64(Normal::new(0., 1.)));

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
        let ten:Elem = 10.0;
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

    let input = Array::random((10, 1, 5, 5), F64(Normal::new(0., 1.)));
    let f = |x: ArrayView4<Elem>| -> Elem {
        let x2 = affine_layer.forward_view(x);
        let loss = softmax_with_loss_layer.forward_view(x2.view(), &answer_array1);
        loss
    };

    let ret = numerical_gradient_array(&input, f);
    assert_eq!(input.shape(), ret.shape());
}


#[test]
fn test_compare_numerical_gradient_array4() {
    let mut affine_layer = Affine::new(25, 10);

    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let input = Array::random((10, 1, 5, 5), F64(Normal::new(0., 1.)));
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    let affine_output = affine_layer.forward(&input);
    let _ = softmax_with_loss_layer.forward(&affine_output, &answer_array1);
    let dout = softmax_with_loss_layer.backward(1.);
    let analytical_gradient = affine_layer.backward(&dout);

    let numerical_gradient = numerical_gradient_array(&input, |x: ArrayView4<Elem>| -> Elem {
        let x2 = affine_layer.forward_view(x);
        let loss = softmax_with_loss_layer.forward_view(x2.view(), &answer_array1);
        loss
    });
    assert_eq!(input.shape(), numerical_gradient.shape());

    let rel_error = relative_error(&numerical_gradient, &analytical_gradient);
    assert_approx_eq!(rel_error, 0., 0.1);
}

#[test]
fn test_norm() {
    // Frobenius norm
    // https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.norm.html
    let input = arr2(&[[-4., -3., -2.],    
               [-1., 0., 1.],
               [2., 3., 4.]]);
    assert_eq!(norm(&input), 7.745966692414834_f64);

    let input = arr2(&[[ 0.54041532, 0.80810253, 0.26378049],
        [ 0.40096443, 0.52327029, 0.84360887],
        [ 0.65346527, 0.16589569, 0.39340066],
        [ 0.37175547, 0.91136225, 0.06099962]]);
    // Numpy shows 1.93461244497
    assert_approx_eq!(norm(&input), 1.93461244497_f64);
}