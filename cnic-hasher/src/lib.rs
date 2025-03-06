const DEFAULT_CAPACITY: usize = 8;
const CNIC_LENGTH: usize = 13;
const LOAD_FACTOR: f64 = 0.6;
const ZERO_ASCII_VALUE: u8 = 0x30;

pub struct CnicHashmap {
    len: usize,
    cap: usize,
    arr: Vec<Option<String>>,
}

impl Default for CnicHashmap {
    fn default() -> Self {
        Self {
            len: 0,
            cap: DEFAULT_CAPACITY,
            arr: vec![None; DEFAULT_CAPACITY],
        }
    }
}

impl CnicHashmap {
    pub fn new() -> Self {
        Self {
            len: 0,
            cap: DEFAULT_CAPACITY,
            arr: vec![None; DEFAULT_CAPACITY],
        }
    }

    pub fn push(&mut self, s: &str) -> i64 {
        if !validate_cnic(s) {
            return -1;
        }

        if self.len as f64 > LOAD_FACTOR * self.cap as f64 {
            self.expand_arr();
        }

        self.len += 1;

        // CHECK: error handling/better way to do this?
        return push_to_map(&mut self.arr, s).try_into().unwrap();
    }

    pub fn remove(&mut self, s: &str) -> bool {
        if !validate_cnic(s) {
            return false;
        }

        for opt in self.arr.iter_mut() {
            match opt {
                Some(entry) => {
                    if entry == s {
                        *opt = None;
                        self.len -= 1;
                        return true;
                    }
                }
                None => continue,
            }
        }

        false
    }

    pub fn print(self) {
        for (idx, opt) in self.arr.iter().enumerate() {
            if let Some(s) = opt {
                println!("Index: {idx}, Value: {s}");
            }
        }
    }

    fn expand_arr(&mut self) {
        self.cap *= 2;
        let mut new_vec: Vec<Option<String>> = vec![None; self.cap];

        self.arr.iter().flatten().for_each(|s| {
            push_to_map(&mut new_vec, &s);
        });

        self.arr = new_vec;
    }
}

fn hash(s: &str) -> usize {
    let middle: usize = s[5..11].parse().unwrap();
    let final_digit: u8 = s.as_bytes()[12] - ZERO_ASCII_VALUE;
    middle << final_digit
}

// Calculates hash and inserts key; arr's length is taken to be the capacity.
fn push_to_map(arr: &mut Vec<Option<String>>, s: &str) -> usize {
    let mut idx: usize = hash(s) % arr.capacity();

    while arr[idx].is_some() {
        idx += 1;
        if idx >= arr.capacity() {
            idx = 0;
        }
    }

    arr[idx] = Some(s.to_owned());
    idx
}

fn validate_cnic(s: &str) -> bool {
    s.starts_with("35202")
        && s.len() == CNIC_LENGTH
        && s[5..].chars().all(|c| c.to_digit(10).is_some())
}
