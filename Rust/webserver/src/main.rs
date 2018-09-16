use std::io::prelude::*;
use std::net::TcpStream;
use std::net::TcpListener;
use std::fs::File;
use std::io;
extern crate webserver;
use webserver::ThreadPool;

// https://doc.rust-lang.org/book/second-edition/ch20-01-single-threaded.html
fn main() {
    let listener = TcpListener::bind("127.0.0.1:7878").unwrap();
    let pool = ThreadPool::new(4);

    for stream in listener.incoming() {
        let stream = stream.unwrap();
        pool.execute(|| {
            let ret = handle_connection(stream);
            match ret {
                Ok(_) => {
                    println!("Connection handled");
                }
                Err(e) => {
                    println!("error handling connection: {:?}", e);
                }
            }
        });
    }
}


fn handle_connection(mut stream: TcpStream) -> io::Result<()> {
    let mut buffer = [0; 512]; // Buffer of 512 bytes in stack
    // .unwrap() works, but ? doens't.
    stream.read(&mut buffer)?;
    let get_root = b"GET / HTTP/1.1\r\n";
    let (status_line, filename) = if buffer.starts_with(get_root) {
        ("HTTP/1.1 200 OK", "hello.html")
    } else {
        ("HTTP/1.1 404 NOT FOUND", "404.html")
    };
    let mut file = File::open(filename).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let response = format!("{}\r\n\r\n{}", status_line, contents);
    stream.write(response.as_bytes()).unwrap();
    stream.flush()
}