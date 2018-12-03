struct NumList {
    numbers1: Vec<f32>,
    numbers2: Vec<f32>,
}

impl NumList {
    fn gradient<F>(&mut self, weights: &mut Vec<f32>, f: F) -> Vec<f32>
    where
        F: Fn(&Vec<f32>) -> f32,
    {
        let h = 0.0001;
        let mut ret = Vec::new();
        for i in 0..weights.len() {
            let original_value = self.numbers[i];

            weights[i] = original_value + h;
            let fx_plus_h = f();

            weights[i] = original_value - h;
            let fx_minus_h = f();

            weights[i] = original_value;
            ret.push((fx_plus_h - fx_minus_h) / 2.);
        }
        ret
    }
}

fn main() {
    let mut num_list = NumList {
        numbers1: vec![1., 2., 3.],
        numbers2: vec![4., 5., 6.],
    };
    let sum_result = num_list.gradient(&mut num_list.numbers1,
    || num_list.numbers1.iter().fold(0., |sum, val| sum + val));
    let product_result = num_list.gradient(|v| v.iter().fold(1., |product, val| product * val));
    println!("results: {:?}, {:?}", sum_result, product_result);
}
