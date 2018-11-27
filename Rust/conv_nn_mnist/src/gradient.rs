use ndarray::prelude::*;

use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::Ix;
use ndarray::Zip;
use ndarray::{arr2, arr3};
use ndarray_rand::{RandomExt, F32};
use num_traits::identities::Zero;
use rand::distributions::Normal;
use std::cmp::max;

use layer::{argmax2d, Affine, Convolution, Elem, Layer, Matrix, Pooling, Relu, SoftmaxWithLoss};

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

fn numerical_gradient_array_h<D, F>(x: &Array<Elem, D>, mut f: F, h: Elem) -> Array<Elem, D>
where
    F: FnMut(ArrayView<Elem, D>) -> Elem,
    D: Dimension,
{
    let mut ret = Array::<Elem, D>::zeros(x.raw_dim());
    let mut x = x.to_owned();
    let x_iter = x.to_owned();
    for (p, i) in x_iter.indexed_iter() {
        // When we cannot borrow x in the loop, what can we do with it?

        // the trait `ndarray::NdIndex<D>` is not implemented for `<D as ndarray::Dimension>::Pattern
        let dim = p.into_dimension();
        let original_element: Elem = *i;

        x[dim.clone()] = original_element + h;
        let fx_plus_h = f(x.view());

        x[dim.clone()] = original_element - h;
        let fx_minus_h = f(x.view());

        x[dim.clone()] = original_element;

        let d = fx_plus_h - fx_minus_h;
        let dx: Elem = d / (2. * h);
        ret[dim] = dx;
    }
    ret
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
fn test_numerical_gradient_array2_different_h() {
    let mut softmax_with_loss_layer_numerical_gradient = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    let input = Array::random((10, 3), F32(Normal::new(0., 1.)));

    let mut softmax_with_loss_layer_analytical_gradient = SoftmaxWithLoss::new();
    let _ = softmax_with_loss_layer_analytical_gradient.forward(&input, &answer_array1);
    let analytical_gradient = softmax_with_loss_layer_analytical_gradient.backward(1.);

    for i in -10..0 {
        let ten = 10.0_f32;
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
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    let input = Array::random((10, 3), F32(Normal::new(0., 1.)));

    let mut softmax_with_loss_layer_analytical_gradient = SoftmaxWithLoss::new();
    let _ = softmax_with_loss_layer_analytical_gradient.forward(&input, &answer_array1);
    let analytical_gradient = softmax_with_loss_layer_analytical_gradient.backward(1.);

    let f = |x: ArrayView2<Elem>| -> Elem {
        let loss = softmax_with_loss_layer_numerical_gradient.forward_view(x, &answer_array1);
        loss
    };
    let numerical_gradient = numerical_gradient_array(&input, f);

    Zip::from(&numerical_gradient)
        .and(&analytical_gradient)
        .apply(|a, b| {
            assert_approx_eq!(*a, *b, 0.05);
        });
}

#[test]
fn test_numerical_gradient_arraysum() {
    let input = Array::random((5, 5), F32(Normal::new(0., 1.)));

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
        let ten = 10.0_f32;
        let h = ten.powi(i);
        let numerical_gradient = numerical_gradient_array_h(&input, f, h);
        assert_eq!(input.shape(), numerical_gradient.shape());

        let mut max_diff: Elem = 0.;
        Zip::from(&numerical_gradient).apply(|a| {
            max_diff = max_diff.max((a - 1.).abs());
        });
        println!("h: {}, maximum diff: {}", h, max_diff);
    }
}

#[test]
fn test_numerical_gradient_array4() {
    let mut affine_layer = Affine::new(25, 10);

    let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    let input = Array::random((10, 1, 5, 5), F32(Normal::new(0., 1.)));
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
    let input = Array::random((10, 1, 5, 5), F32(Normal::new(0., 1.)));
    let answer_array1 = Array1::from_vec(vec![0, 1, 2, 1, 1, 0, 1, 1, 2, 1]);

    let affine_output = affine_layer.forward(&input);
    let _ = softmax_with_loss_layer.forward(&affine_output, &answer_array1);
    let dout = softmax_with_loss_layer.backward(1.);
    let analytical_gradient = affine_layer.backward(&dout);

    let f = |x: ArrayView4<Elem>| -> Elem {
        let x2 = affine_layer.forward_view(x);
        let loss = softmax_with_loss_layer.forward_view(x2.view(), &answer_array1);
        loss
    };


    let numerical_gradient = numerical_gradient_array(&input, f);
    assert_eq!(input.shape(), numerical_gradient.shape());

    let mut max_diff: Elem = 0.;

    Zip::from(&numerical_gradient)
        .and(&analytical_gradient)
        .apply(|a, b| {
            max_diff = max_diff.max((*a - *b).abs());
//            assert_approx_eq!(*a, *b, 0.05);
        });
    // As of Nov 26th, max_diff: 0.19 - 0.32. Is this enough?
    println!("max_diff: {}", max_diff);
}
