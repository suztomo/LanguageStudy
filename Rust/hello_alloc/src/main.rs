extern crate stats_alloc;

use stats_alloc::{StatsAlloc, Region, INSTRUMENTED_SYSTEM};
use std::alloc::System;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn example_using_region() {
    let reg = Region::new(&GLOBAL);
    let mut x: Vec<i32> = Vec::with_capacity(256);
    let k = 34;
    for i in 0..257 {
        x.push(i);
    }
    println!("Stats at 1: {:#?}", reg.change());
    /* Example output of this:
    Stats at 1: Stats {
    allocations: 1,
    deallocations: 0,
    reallocations: 6,
    bytes_allocated: 65536,
    bytes_deallocated: 0,
    bytes_reallocated: 64512
}
}*/

    ::std::mem::size_of_val(&x) + k;
}


fn main() {
    let k = example_using_region();
    println!("Hello, world! {:?}", k);
}
