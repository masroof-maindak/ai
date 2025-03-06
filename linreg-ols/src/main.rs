use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
struct Entry {
    #[serde(rename = "X1")]
    x1: f64,
    y: f64,
}

fn main() -> Result<(), csv::Error> {
    // Load CSV file
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path("./dataset.csv")?;

    // Deserialize into vector
    let mut dataset: Vec<Entry> = Vec::with_capacity(1000);
    dataset.extend(reader.deserialize().flatten());

    Ok(())
}
