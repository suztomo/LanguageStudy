fn main() {
    let mut v = vec![1, 2, 3];
    for iter in v.iter_mut() {
        *iter *= 2;
    }
    println!("vector {}", v[0]);
}
