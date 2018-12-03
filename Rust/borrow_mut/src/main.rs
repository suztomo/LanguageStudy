struct NumList {
    numbers: Vec<f32>,
}

impl NumList {
    fn gradient<F>(&mut self, f: F) -> Vec<f32>
    where
        F: Fn(&Vec<f32>) -> f32,
    {
        let h = 0.0001;
        let mut ret = Vec::new();
        for n in self.numbers.iter_mut() {
            let original_value = *n;

            *n = original_value + h;
            let fx_plus_h = f(&self.numbers);

            *n = original_value - h;
            let fx_minus_h = f(&self.numbers);

            *n = original_value;
            ret.push((fx_plus_h - fx_minus_h) / 2.);
        }
        ret
    }
}

fn main() {
    let mut num_list = NumList {
        numbers: vec![1., 2., 3.],
    };
    let sum_result = num_list.gradient(|v| v.iter().fold(0., |sum, val| sum + val));
    let product_result = num_list.gradient(|v| v.iter().fold(1., |product, val| product * val));
    println!("results: {:?}, {:?}", sum_result, product_result);
}
