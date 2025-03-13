use csv::ReaderBuilder;
use plotters::prelude::*;
use serde::Deserialize;

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
struct Point {
    #[serde(rename = "X1")]
    x: f64,
    y: f64,
}

fn compute_regression_coefficients(dataset: &Vec<Point>) -> (f64, f64) {
    let x_vec: Vec<f64> = dataset.iter().map(|e| e.x).collect();
    let y_vec: Vec<f64> = dataset.iter().map(|e| e.y).collect();

    let x_mean: f64 = x_vec.iter().sum::<f64>() / x_vec.len() as f64;
    let y_mean: f64 = y_vec.iter().sum::<f64>() / y_vec.len() as f64;

    // Sum{ (x - x_mean) * (y - y_mean) }
    let numerator = dataset
        .iter()
        .map(|e| (e.x - x_mean) * (e.y - y_mean))
        .sum::<f64>();

    // Sum{ (x - x_mean)^2 }
    let denominator = x_vec.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();

    let best_fit_slope = numerator / denominator;
    let intercept = y_mean - (best_fit_slope * x_mean);

    (best_fit_slope, intercept)
}

fn compute_mean_squared_error(dataset: &Vec<Point>, best_fit_slope: f64, intercept: f64) -> f64 {
    dataset
        .iter()
        .map(|e| (e.y - (best_fit_slope * e.x + intercept)).powi(2))
        .sum::<f64>()
        / dataset.len() as f64
}

fn normalize_features(dataset: &mut Vec<Point>) {
    let (x_min, x_max, _, _) = get_min_max(&dataset);
    for e in dataset.iter_mut() {
        e.x = ((e.x - x_min) / (x_max - x_min)) as f64;
    }
}

fn split_dataset(
    dataset: &Vec<Point>,
    split_perc: f32,
) -> Result<(Vec<Point>, Vec<Point>, usize), &'static str> {
    if split_perc < 0.0 || split_perc > 1.0 {
        return Err("invalid split percentage".into());
    }

    let cutoff_idx = (split_perc * dataset.len() as f32) as usize;

    let train_set = dataset.iter().take(cutoff_idx).map(|e| e.clone()).collect();
    let test_set = dataset.iter().skip(cutoff_idx).map(|e| e.clone()).collect();
    Ok((train_set, test_set, cutoff_idx))
}

fn get_min_max(dataset: &Vec<Point>) -> (f64, f64, f64, f64) {
    let mut x_min = dataset[0].x;
    let mut x_max = dataset[0].x;
    let mut y_min = dataset[0].y;
    let mut y_max = dataset[0].y;

    for point in dataset.iter() {
        if point.x < x_min {
            x_min = point.x;
        }
        if point.x > x_max {
            x_max = point.x;
        }
        if point.y < y_min {
            y_min = point.y;
        }
        if point.y > y_max {
            y_max = point.y;
        }
    }

    (x_min, x_max, y_min, y_max)
}

fn plot_regression_and_points(
    dataset: &Vec<Point>,
    cutoff_idx: usize,
    line: f64,
    intercept: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let (x_min, x_max, y_min, y_max) = get_min_max(dataset);

    let root = BitMapBackend::new("output.png", (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Linear Regression w/ Min-Max Scaling",
            ("IBM Plex Sans", 48).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(dataset.iter().enumerate().map(|(i, p)| {
        if i < cutoff_idx {
            Circle::new((p.x, p.y), 3, BLUE.filled())
        } else {
            Circle::new((p.x, p.y), 3, YELLOW.filled())
        }
    }))?;

    let regression_line = vec![
        (x_min, line * x_min + intercept),
        (x_max, line * x_max + intercept),
    ];

    chart.draw_series(LineSeries::new(regression_line, &RED))?;

    root.present()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path("./dataset01.csv")?;

    let mut dataset: Vec<Point> = Vec::with_capacity(1000);
    dataset.extend(reader.deserialize().flatten());

    let split_perc = 0.8;

    // Normalize & Split
    normalize_features(&mut dataset);
    let (train_set, test_set, cutoff_idx) = split_dataset(&dataset, split_perc)?;

    let (m, c) = compute_regression_coefficients(&train_set);
    let mse = compute_mean_squared_error(&test_set, m, c);
    plot_regression_and_points(&dataset, cutoff_idx, m, c)?;

    println!("Plot saved as output.png");
    println!("Mean Squared Error: {mse}");
    Ok(())
}
