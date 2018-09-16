#[macro_use]
extern crate log;
extern crate env_logger;

fn largest_copy0<T: PartialOrd + Copy>(lst: &[T]) -> T {
    let mut largest = lst[0];
    for item in lst.iter() {
        if item > &largest {
            largest = *item
        }
    }
    largest
}
fn largest_copy1<T: PartialOrd + Copy>(lst: &[T]) -> T {
    let mut largest = lst[0];
    for &item in lst.iter() {
        if item > largest {
            largest = item
        }
    }
    largest
}

fn largest_ref<T: PartialOrd>(lst: &[T]) -> &T {
    let mut largest: &T = &lst[0];
    for item in lst.iter() {
        if item > largest {
            largest = item
        }
    }
    largest
}

// ~/Documents/.../Rust/find_largest $ cargo run
fn main() {
    // Environment variable with RUST_LOG=info does the trick
    env_logger::init();
    let lst = vec![23, 43, 23, 3, 34];
    let l = largest_copy1(&lst);
    info!("Largest from copy {}", l);
    let m = largest_ref(&lst);
    info!("Largest from ref {}", m);
    println!("l {}, m {}", l, m);
}
