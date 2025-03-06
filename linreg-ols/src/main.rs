use csv::ReaderBuilder;
use plotters::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "type")]
struct Point {
    #[serde(rename = "X1")]
    x: f64,
    y: f64,
}

fn ordinary_least_squares(dataset: &Vec<Point>) -> (f64, f64) {
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
    line: f64,
    intercept: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output.png", (1600, 1200)).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max, y_min, y_max) = get_min_max(dataset);

    let mut chart = ChartBuilder::on(&root)
        .caption("Linear Regression", ("IBM Plex Sans", 48).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        dataset
            .iter()
            .map(|p| Circle::new((p.x, p.y), 3, BLUE.filled())),
    )?;

    let regression_line = vec![
        (x_min, line * x_min + intercept),
        (x_max, line * x_max + intercept),
    ];

    chart.draw_series(LineSeries::new(regression_line, &RED))?;

    root.present()?;

    Ok(())
}

fn main() -> Result<(), csv::Error> {
    // Load CSV file
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path("./dataset.csv")?;

    // Deserialize to vector
    let mut dataset: Vec<Point> = Vec::with_capacity(1000);
    dataset.extend(reader.deserialize().flatten());

    let (b, c) = ordinary_least_squares(&dataset);
    let mse = compute_mean_squared_error(&dataset, b, c);

    if let Err(e) = plot_regression_and_points(&dataset, b, c) {
        eprintln!("Error while plotting data: {e}");
    } else {
        println!("Plot saved as output.png");
    }

    println!("Mean Squared Error: {mse}");
    Ok(())
}
