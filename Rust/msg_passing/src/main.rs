use std::thread;
use std::sync::mpsc;
// https://doc.rust-lang.org/book/second-edition/ch16-02-message-passing.html
fn main() {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let val = "hi";
        tx.send(val).unwrap();
        println!("child thread sent string.");
        tx.send("foo").unwrap();
        println!("child thread sent string.");
    });

    let received_str = rx.recv().unwrap();
    println!("Main thread got: {}", received_str);
    let received_int = rx.recv().unwrap();
    println!("Main thread got: {}", received_int);
}
