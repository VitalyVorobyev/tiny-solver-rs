// Spike sweep: random robust fits (line and exponential), LM and GN with tight tolerances.
// Prints: seed model loss solver sum_rho params...
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nalgebra as na;
use tiny_solver::loss_functions::{ArctanLoss, CauchyLoss, HuberLoss, Loss};
use tiny_solver::optimizer::{Optimizer, OptimizerOptions};
use tiny_solver::{GaussNewtonOptimizer, LevenbergMarquardtOptimizer};

struct Fit {
    evaluations: Arc<AtomicUsize>,
    model: usize,
    x: f64,
    y: f64,
}
impl<T: na::RealField> tiny_solver::factors::Factor<T> for Fit {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let p = &params[0];
        let x = T::from_f64(self.x).unwrap();
        let y = T::from_f64(self.y).unwrap();
        let f = if self.model == 0 {
            p[0].clone() * x + p[1].clone()
        } else {
            p[0].clone() * (-(p[1].clone() * x)).exp() + p[2].clone()
        };
        na::dvector![f - y]
    }
}

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.u()
    }
    fn g(&mut self) -> f64 {
        (0..12).map(|_| self.u()).sum::<f64>() - 6.0
    }
}

fn rho(loss: usize, a: f64, s: f64) -> f64 {
    match loss {
        0 => if s > a * a { 2.0 * a * s.sqrt() - a * a } else { s },
        1 => a * a * (1.0 + s / (a * a)).ln(),
        _ => a * s.atan2(a),
    }
}

fn main() {
    env_logger::init();
    let n_seeds: u64 = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(200);
    let tight = std::env::args().nth(2).as_deref() == Some("tight");
    let options = if tight {
        OptimizerOptions {
            max_iteration: 500,
            min_abs_error_decrease_threshold: 0.0,
            min_rel_error_decrease_threshold: 0.0,
            min_error_threshold: 0.0,
            ..Default::default()
        }
    } else {
        OptimizerOptions::default()
    };
    let only: Option<u64> = std::env::var("ONLY").ok().map(|s| s.parse().unwrap());
    for seed in 0..n_seeds {
        if only.is_some_and(|o| o != seed) { continue; }
        let mut rng = Lcg(seed * 7919 + 1);
        for _ in 0..3 { rng.u(); }
        let model = (seed % 2) as usize;
        let n = 20 + (rng.u() * 60.0) as usize;
        let frac = rng.range(0.1, 0.45);
        let sigma = rng.range(0.05, 0.5);
        let truth = if model == 0 { vec![rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)] }
                    else { vec![rng.range(1.0, 5.0), rng.range(0.2, 1.5), rng.range(-1.0, 1.0)] };
        let pts: Vec<(f64, f64)> = (0..n).map(|i| {
            let x = i as f64 * 5.0 / n as f64;
            let mut y = if model == 0 { truth[0] * x + truth[1] } else { truth[0] * (-truth[1] * x).exp() + truth[2] };
            y += sigma * rng.g();
            if rng.u() < frac { y += rng.range(2.0, 50.0) * if rng.u() < 0.5 { -1.0 } else { 1.0 }; }
            (x, y)
        }).collect();
        let init: Vec<f64> = truth.iter().map(|t| t + rng.range(-2.0, 2.0)).collect();
        let scale = rng.range(0.1, 2.0);
        for loss in 0..3 {
            if std::env::var("DUMP").is_ok() {
                let a = if loss == 2 { scale * scale } else { scale };
                let i: Vec<String> = init.iter().map(|v| format!("{v:.17e}")).collect();
                let xy: Vec<String> = pts.iter().map(|(x, y)| format!("{x:.17e} {y:.17e}")).collect();
                eprintln!("{seed} {model} {loss} {a:.17e} {} {} {} {}", init.len(), i.join(" "), pts.len(), xy.join(" "));
            }
            let evaluations = Arc::new(AtomicUsize::new(0));
            let mut problem = tiny_solver::Problem::new();
            for &(x, y) in &pts {
                let l: Box<dyn Loss + Send> = match loss {
                    0 => Box::new(HuberLoss::new(scale)),
                    1 => Box::new(CauchyLoss::new(scale)),
                    _ => Box::new(ArctanLoss::new(scale * scale)),
                };
                let l = if std::env::var("NOLOSS").is_ok() { None } else { Some(l) };
                problem.add_residual_block(1, &["p"], Box::new(Fit { evaluations: evaluations.clone(), model, x, y }), l);
            }
            let initial = HashMap::from([("p".to_string(), na::DVector::from_vec(init.clone()))]);
            let a = if loss == 2 { scale * scale } else { scale };
            for (name, opt) in [("lm", Box::new(LevenbergMarquardtOptimizer::default()) as Box<dyn Optimizer>),
                                ("gn", Box::new(GaussNewtonOptimizer::new()))] {
                evaluations.store(0, Ordering::Relaxed);
                let start = Instant::now();
                let result = opt.optimize(&problem, &initial, Some(options.clone()));
                let micros = start.elapsed().as_secs_f64() * 1e6;
                let evals = evaluations.load(Ordering::Relaxed) as f64 / pts.len() as f64;
                match result {
                    Some(r) => {
                        let p = &r["p"];
                        let cost: f64 = pts.iter().map(|&(x, y)| {
                            let f = if model == 0 { p[0] * x + p[1] } else { p[0] * (-p[1] * x).exp() + p[2] };
                            rho(loss, a, (f - y).powi(2))
                        }).sum();
                        let ps: Vec<String> = p.iter().map(|v| format!("{v:.10}")).collect();
                        println!("{seed} {model} {loss} {name} {cost:.12e} {} evals={evals:.0} us={micros:.0}", ps.join(" "));
                    }
                    None => println!("{seed} {model} {loss} {name} fail evals={evals:.0} us={micros:.0}"),
                }
            }
        }
    }
}
