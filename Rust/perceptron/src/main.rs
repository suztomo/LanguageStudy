#[macro_use]
extern crate ndarray;
//extern crate ndarray_linalg;
extern crate rand;
use rand::Rng;
use ndarray::prelude::*;
//use ndarray_linalg::Eigh;
//use ndarray_linalg::lapack_traits::UPLO;

/*
  Let's learn Rust by implementing simple perceptron
 https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
*/

// NDarray for numpy users
// https://docs.rs/ndarray/0.11.2/ndarray/doc/ndarray_for_numpy_users/index.html



// ld: library not found for -lgfortran
// brew install gfortran -> no such library
// Trying to open XCode, hoping that it'll install gfortran by default
//   'ran xcode-select --install' as instructed in readme of gfortran https://gcc.gnu.org/wiki/GFortranBinaries#MacOS

fn main() {
    let training_data: Array2<f32> = arr2(&[
        [0., 0., 1.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.]
        ]);
    let ans = vec![0., 0., 0., 1.];
    let unit_step = |i: f32| -> f32 {
        if i < 0. {
            0.
        } else {
            1.
        }
    };
    //let (e, vecs) = a.clone().eigh(UPLO::Upper).unwrap(); // 固有値計算(エルミート)
    //println!("eigenvalues = \n{:?}", e);
    //println!("V = \n{:?}", vecs);
    //let av = a.dot(&vecs);
    //println!("AV = \n{:?}", av);

    // >>> random.rand(3)
    // array([ 0.07981295,  0.34492088,  0.46878779])
    let mut w: Array<f32, Dim<[usize; 1]>> = arr1(&[0.11, -0.3, 0.1]); // random
/*
    let mut w1: Array<f32, Dim<[usize; 1]>> = arr1(&[0.5, 0.3, 0.1]); // random
    let mut w2: Array<f32, Dim<[usize; 1]>> = arr1(&[0.5, 0.3, 0.1]); // random
    let v1: ArrayView<f32, Dim<[usize; 1]>> = w1.slice(s![0..]);
    let v2: ArrayView<f32, Dim<[usize; 1]>> = w2.slice(s![0..]);

    let k = w2 - w1;
    println!("k = {:?}", k);
    */
    // ArrayView doesn't allow me to use '-' operator
    //let l = v2 - v1;
    //println!("l = {:?}", l);

    let mut errors:Vec<f32> = Vec::new();
    let n = 1000;
    let eta = 0.2;
    for _ in 1..n {
        let choice: usize = rand::thread_rng().gen_range(0, ans.len());
        // ndarray 0.11.0 has: Add support for more index types (e.g. usize) to the s![] macro by @jturner314
        let x:ArrayView1<f32> = training_data.slice(s![choice, ..]);
        let expected = &ans[choice];
        // It looks like w doesn't needs to be horizontal
        let res = x.dot(&w);
        //println!("v {:?} dot w {:?} = {}", &x, &w, res);
        let err = expected - unit_step(res);
        errors.push(err);
        // ArrayView doesn't allow me to use '-' operation
        // Should I avoid ArrayView?
        //let w_mod = w.sub(&x);
        //w.assign(&w_mod);// * eta * err
        calc(&mut w, x, eta, err);
    }
    //  [4, 3]
    let l = training_data.shape();
    for i in 0..l[0] {
        let tdata:ArrayView1<f32> = training_data.slice(s![i,..]);
        let result = tdata.dot(&w);
        println!("{:?} = {} -> {}", tdata, result, unit_step(result));
    }
//    for x, _ in training_data: result = dot(x, w)
// print("{}: {} -> {}".format(x[:2], result, unit_step(result)))
}

fn calc(x:&mut Array1<f32>, y:ArrayView1<f32>, eta: f32, err: f32) {
    for (mut lhs, rhs) in x.iter_mut().zip(y.iter()) {
        *lhs += eta * err * rhs;
    }
}
