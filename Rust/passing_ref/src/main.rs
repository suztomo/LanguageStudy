fn add_php(list: &mut Vec<String>) {
    let list = &mut Vec::new();
    list.push(String::from("PHP"));
}

fn add_int(a: i32) {
    // This doesn't compile "cannot assign twice to immutable variable"
    //a = 5;
    println!("a is {}", a);
}

fn main() {
    let mut list = vec![String::from("Rust")];
    add_php(&mut list);
    println!("list is {:?}", list);
}
