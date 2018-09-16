
type Job = Box<FnOnce() + Send + 'static>;

struct FooFunc<'a> {
    func: &'a dyn (FnOnce())
}


//fn run_foo_func(ffunc: &FooFunc) {
    // error[E0161]: cannot move a value of type dyn std::ops::FnOnce(): the size of dyn std::ops::FnOnce() cannot be statically determined

    // Right, closure cannot tell how much the value will consume memory. So the size is unknown.
    // "dyn", which is introduced in Rust 1.27, explicitly specifies the following item is trait, not type
    // because trait doesn't know the size of the memory allocation, the compiler complains

    // Then, where does the value (that implements the trait) is being moved?
//    (ffunc.func)();
//}

struct FooStr {
    s: String
}

fn fun_foo_str(fstr: &FooStr) {
    // the size for values of type `str` cannot be known at compilation time
    //let k = fstr.s;
    //println!("fstr is {}", k);
}
struct FooString {
    s: String
}

fn fun_foo_string(fstr: &FooString) {
    // cannot move out of borrowed content
    // having &fstr.s solves the issue.
    //let k = fstr.s;
    //println!("fstr is {}", k);
}

fn run_job(f: Job) {
    // This complains with below
    // cannot move a value of type (dyn std::ops::FnOnce() + std::marker::Send + 'static): the size of (dyn std::ops::FnOnce() + std::marker::Send + 'static) cannot be statically determined
    // --> src/main.rs:5:5
    //  |
    //5 |     f();
    //  |     ^
    // So calling a closure is moving something?

    // > the closure needs to move itself out of the Box<T> because the closure takes ownership of self when we call it.

    // "rustc --explain E0161" explains that
    // In Rust, you can only move a value when its size is known at compile time.
    //(*f)();
}

// The below fails with
// = help: the trait `std::marker::Sized` is not implemented for `(dyn std::ops::FnOnce() + 'static)`
// = note: to learn more, visit <https://doc.rust-lang.org/book/second-edition/ch19-04-advanced-types.html#dynamically-sized-types--sized>
//fn run_closure(closure: FnOnce()) {
//    closure();
//}
//fn run_closure_ref(closure: &FnOnce()) {
//    closure();
//}
//fn run_closure_box(closure: Box<FnOnce()>) {
//    (*closure)();
//}

fn main() {
    // https://doc.rust-lang.org/book/second-edition/ch20-02-multithreaded.html
    // "This error is fairly cryptic because the problem is fairly cryptic."
    println!("Hello, world!");
    let f = || {
            println!("This is closure");
        };
    let job_box = Box::new(f);


    // The code below for g fails:
    // note: to learn more, visit <https://doc.rust-lang.org/book/second-edition/ch19-04-advanced-types.html#dynamically-sized-types--sized>
    // = note: all local variables must have a statically known size
    let g:&FnOnce() = & (|| {
            println!("This is second closure");
        });
    let ffunc = FooFunc { func : g};

    //jobBox();// works fine
    // This doesn't work
    //run_job(jobBox);
    //run_closure_ref(g);
    //let k: Box<FnOnce()> = Box::new(g);
    //run_closure_box(jobBox);
    (*job_box)();

    let k = FooString{ s: String::from("abc") };
    fun_foo_string(&k);
}
