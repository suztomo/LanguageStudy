use ndarray::prelude::*;
use ndarray::Dimension;
use ndarray::IntoDimension;
use layer::{argmax2d, Affine, Convolution, Elem, Layer, Matrix, Pooling, Relu, SoftmaxWithLoss, Loss, mnist_to_nchw};
use mnist::{Grayscale, MnistRecord, IMG_H_SIZE, IMG_W_SIZE, Label};

pub trait Network<'a> {
    fn train(&mut self, minibatch: Matrix, answers: Vec<Label>, learning_rate: f64) -> Loss;
    fn forward_path_before_loss(&mut self, minibatch: Matrix) -> Array2<Elem>;
    fn forward_path(&mut self, minibatch: Matrix, answers: Vec<Label>) -> Loss;
    fn backward_path(&mut self) -> Matrix;
    fn predict(&mut self, mnist: &MnistRecord) -> Label;
    fn weights_ref(&mut self) -> (Vec<(String, &mut Array2<Elem>)>, Vec<(String, &mut Array1<Elem>)>);
    fn numerical_gradient(&mut self, input: Matrix, answers: Vec<Label>) -> (Vec<(String, Array1<Elem>)>, Vec<(String, Array2<Elem>)>);
}

#[derive(Debug)]
pub struct SimpleConv {
    convolution_layer: Convolution,
    relu_layer: Relu<Ix4>,
    pooling_layer: Pooling,
    affine_layer: Affine,
    relu2_layer: Relu<Ix2>,
    affine2_layer: Affine,
    softmax_layer: SoftmaxWithLoss
}

impl SimpleConv {
    pub fn new(batch_size: usize, img_size:(usize, usize)) -> SimpleConv {
        let padding = 0; // To make 24x24 to 28x28
        let affine1_output_size = 100;
        let class_num = 10;
        let conv_filter_num = 30;
        let conv_channel_num = 1;
        let convolution_layer = Convolution::new(batch_size, conv_filter_num, conv_channel_num, 5, 5, 1, padding);
        let (conv_output_h, conv_output_w) = convolution_layer.output_size(img_size.0, img_size.1);
        let pooling_layer = Pooling::new(2, 2, 2, 0);
        let (pooling_output_h, pooling_output_w) = pooling_layer.output_size(conv_output_h, conv_output_w);
        let pooling_output_size = conv_filter_num * conv_channel_num * pooling_output_h * pooling_output_w;
        SimpleConv {
            convolution_layer,
            relu_layer : Relu::<Ix4>::new(),
            pooling_layer,
            affine_layer : Affine::new(pooling_output_size, affine1_output_size),
            relu2_layer : Relu::<Ix2>::new(),
            affine2_layer : Affine::new(affine1_output_size, class_num),
            softmax_layer : SoftmaxWithLoss::new(),
        }
    }
    /*
    fn each_weights_ref<F, D>(&mut self,  mut loss_f: F) -> Vec<Array<Elem, D>> where
    F: FnMut() -> f64,
    D: Dimension, {
        let gradient_affine_weights = numerical_gradient_weights(&mut self.affine_layer.weights, loss_f);
        vec![gradient_affine_weights]
    }*/
}



impl<'a> Network<'a> for SimpleConv {
    fn forward_path_before_loss(&mut self, minibatch: Matrix) -> Array2<Elem> {
        let conv_output = self.convolution_layer.forward(&minibatch);
        let relu_output = self.relu_layer.forward(&conv_output);
        let pooling_output = self.pooling_layer.forward(&relu_output);
        let affine_output = self.affine_layer.forward(&pooling_output);
        let relu2_output = self.relu2_layer.forward(&affine_output);
        self.affine2_layer.forward_2d(&relu2_output)
    }

    fn forward_path(&mut self, minibatch: Matrix, answers: Vec<Label>) -> Loss {        
        let affine2_output = self.forward_path_before_loss(minibatch);
        let softmax_output = self.softmax_layer.forward(&affine2_output, &Array1::from_vec(answers));
        softmax_output as Loss
    }

    fn backward_path(&mut self) -> Matrix {
        let softmax_dx = self.softmax_layer.backward(1.);
        let affine2_dx = self.affine2_layer.backward_2d(&softmax_dx);
        let relu2_dx = self.relu2_layer.backward(&affine2_dx);
        let affine_dx = self.affine_layer.backward(&relu2_dx);
        let pooling_dx = self.pooling_layer.backward(&affine_dx);
        let relu_dx = self.relu_layer.backward(&pooling_dx);
        self.convolution_layer.backward(&relu_dx)
    }

    fn train(&mut self, minibatch: Matrix, answers: Vec<Label>, learning_rate: f64) -> Loss {
        let loss = self.forward_path(minibatch, answers);
        self.backward_path();

        self.affine2_layer.weights -= &(&self.affine2_layer.d_weights * learning_rate);
        self.affine2_layer.bias -= &(&self.affine2_layer.d_bias * learning_rate);

        self.affine_layer.weights -= &(&self.affine_layer.d_weights * learning_rate);
        self.affine_layer.bias -= &(&self.affine_layer.d_bias * learning_rate);

        self.convolution_layer.weights -= &(&self.convolution_layer.d_weights * learning_rate);
        self.convolution_layer.bias -= &(&self.convolution_layer.d_bias * learning_rate);
        loss
    }

    fn predict(&mut self, mnist: &MnistRecord) -> Label {
        let nchw = mnist_to_nchw(&mnist);
        let affine2_output = self.forward_path_before_loss(nchw);
        let affine2_output_argmax = argmax2d(&affine2_output, Axis(1));
        let predicted_label = affine2_output_argmax[[0]];
        predicted_label as Label
    }
    fn weights_ref(&mut self) -> (Vec<(String, &mut Array2<Elem>)>, Vec<(String, &mut Array1<Elem>)>) {
        (vec![(String::from("affine1 weights"), &mut self.affine_layer.weights)],
         vec![(String::from("affine2 bias"),  &mut self.affine2_layer.bias)])
    }

    fn numerical_gradient(&mut self, input: Matrix, answers: Vec<Label>) -> (Vec<(String, Array1<Elem>)>, Vec<(String, Array2<Elem>)>) {
        // I was thinking to create generic through (&mut) weights and
        // forward_path() method. This would generate easy access to 
        // gradient check as long as the network implements list of weights inside
        // the network and forward_path would be called with the weights is
        // changed a little bit through the mutable references of weights.
        // The return value of this function would be list of names for bias (Array1)
        // and the list of names for weights (Array2).
        // However, the attempt did not work with Rust's borrow checker.
        // The reason was because the idea requires to borrow the weights mutablly and
        // run forward_path() method while keeping the mutable reference. Rust's
        // borrow checker does not allow us to borrow a reference of an object
        // (in this case `self`) while there is a mutable reference of it within
        // a scope.
        
        // error[E0499]: cannot borrow `*self` as mutable more than once at a time
        //    --> src/network.rs:143:29
        //     |
        // 133 |         let mut weights = &mut self.affine_layer.weights;
        //     |                                ------------------------- first mutable borrow occurs here
        // ...
        // 143 |             let fx_plus_h = self.forward_path(input.to_owned(), answers.to_owned());
        //     |                             ^^^^ second mutable borrow occurs here
        // ...
        // 162 |     }
        //     |     - first borrow ends here
    
        let mut array1_vec: Vec<(String, Array1<Elem>)> = Vec::new();
        let mut array2_vec: Vec<(String, Array2<Elem>)> = Vec::new();
        
        let mut weights = &mut self.affine_layer.weights;

        let h = 0.0001;
        let mut numerical_gradient_affine_weights = Array::zeros(weights.raw_dim());
        let weights_iter = weights.to_owned();
        for (p, _i) in weights_iter.indexed_iter() {
            let dim = p.into_dimension();
            let original_element: Elem = weights[dim.clone()];

            weights[dim.clone()] = original_element + h;
            let fx_plus_h = self.forward_path(input.to_owned(), answers.to_owned());
            
            weights[dim.clone()] = original_element - h;
            let fx_minus_h = self.forward_path(input.to_owned(), answers.to_owned());

            weights[dim.clone()] = original_element;

            let d = fx_plus_h - fx_minus_h;
            let dx = d / (2. * h);
            numerical_gradient_affine_weights[dim] = dx;
        }        
        
        //let numerical_gradient_affine_weights = numerical_gradient_weights(&mut self.affine_layer.weights, || -> Elem {
        //let l = self.forward_path(input.to_owned(), answers.to_owned());
        //println!("loss :{}", l);
        //l
        //});
        array2_vec.push((String::from("numerical_gradient_affine_weights"), numerical_gradient_affine_weights));
        (array1_vec, array2_vec)
    }

}


fn numerical_gradient_weights<D, F>(weights: &mut Array<Elem, D>, mut loss_f: F) -> Array<Elem, D>
where
    F: FnMut() -> f64,
    D: Dimension,
{
    let h = 0.0001;
    let mut ret = Array::<Elem, D>::zeros(weights.raw_dim());
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