use std::collections::HashMap;

use tiny_solver::loss_functions::HuberLoss;
use tiny_solver::{Optimizer, na};

/// r_i = a * x_i + b - y_i
struct LineFactor {
    x: f64,
    y: f64,
}
impl<T: na::RealField> tiny_solver::factors::Factor<T> for LineFactor {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        let (x, y) = (T::from_f64(self.x).unwrap(), T::from_f64(self.y).unwrap());
        na::dvector![params[0][0].clone() * x + params[0][1].clone() - y]
    }
}

fn main() {
    // y = 2x + 1 plus noise; every fourth point is an outlier.
    let points: Vec<(f64, f64)> = (0..12)
        .map(|i| {
            let x = i as f64 * 0.5;
            let noise = 0.5 * (1.7 * i as f64 + 0.3).sin();
            let outlier = if i % 4 == 0 { 10.0 } else { 0.0 };
            (x, 2.0 * x + 1.0 + noise + outlier)
        })
        .collect();
    let mut problem = tiny_solver::Problem::new();
    for &(x, y) in &points {
        let loss = HuberLoss::new(0.25);
        problem.add_residual_block(1, &["ab"], Box::new(LineFactor { x, y }), Some(Box::new(loss)));
    }
    let init = HashMap::from([("ab".to_string(), na::dvector![0.0, 0.0])]);

    let ab = tiny_solver::LevenbergMarquardtOptimizer::default()
        .optimize(&problem, &init, None)
        .unwrap()["ab"]
        .clone();

    // sum_i rho(r_i^2) for Huber with scale 0.25, and its gradient.
    let (mut cost, mut grad) = (0.0, [0.0, 0.0]);
    for &(x, y) in &points {
        let r: f64 = ab[0] * x + ab[1] - y;
        let s = r * r;
        let (rho, rho1) = if s <= 0.0625 { (s, 1.0) } else { (0.5 * s.sqrt() - 0.0625, 0.25 / s.sqrt()) };
        cost += rho;
        grad[0] += 2.0 * rho1 * r * x;
        grad[1] += 2.0 * rho1 * r;
    }
    println!("a = {:.6}, b = {:.6}, sum rho = {cost:.6}, gradient = [{:.2e}, {:.2e}]", ab[0], ab[1], grad[0], grad[1]);
}
