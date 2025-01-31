use cnic_hasher::CnicHashmap;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut ch: CnicHashmap = Default::default();

    for i in &args[1..args.len()] {
        ch.push(i);
    }

    ch.print();
}
