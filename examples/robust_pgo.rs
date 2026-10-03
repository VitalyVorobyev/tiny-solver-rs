// Evaluation harness (not for upstream): g2o pose graphs with a chosen loss and
// optionally injected wrong loop closures. Everything reported is computed here,
// independently of the library's own cost: sum_rho from the factors' plain
// residuals and our own rho, and the gradient of sum_rho via the public
// compute_residual_and_jacobian (J~^T r~ = rho' J^T r for every loss).
//
// usage: robust_pgo <file> <none|huber|cauchy> <scale> <gn|lm> <default|tight>
//                   <outlier_fraction> <seed> <repeats>
use std::collections::HashMap;
use std::fs::read_to_string;
use std::sync::Arc;
use std::time::Instant;

use nalgebra as na;
use tiny_solver::factors::{BetweenFactorSE2, BetweenFactorSE3, Factor, PriorFactor};
use tiny_solver::loss_functions::{CauchyLoss, HuberLoss, Loss};
use tiny_solver::manifold::se3::SE3Manifold;
use tiny_solver::optimizer::{Optimizer, OptimizerOptions};
use tiny_solver::{GaussNewtonOptimizer, LevenbergMarquardtOptimizer, Problem};

#[derive(Clone)]
enum Edge {
    Se2(BetweenFactorSE2),
    Se3(BetweenFactorSE3),
    Prior(PriorFactor),
}

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn rho(loss: &str, a: f64, s: f64) -> f64 {
    match loss {
        "huber" => if s > a * a { 2.0 * a * s.sqrt() - a * a } else { s },
        "cauchy" => a * a * (1.0 + s / (a * a)).ln(),
        _ => s,
    }
}

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let (file, loss, scale, solver, tol) = (&args[1], args[2].as_str(), args[3].parse::<f64>().unwrap(), args[4].as_str(), args[5].as_str());
    let outlier_fraction: f64 = args[6].parse().unwrap();
    let seed: u64 = args[7].parse().unwrap();
    let repeats: usize = args[8].parse().unwrap();

    let mut init = HashMap::<String, na::DVector<f64>>::new();
    let mut edges: Vec<(Edge, Vec<String>)> = Vec::new();
    let mut is_3d = false;
    let mut vertex_ids: Vec<usize> = Vec::new();
    for line in read_to_string(format!("tests/data/{file}")).unwrap().lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.is_empty() { continue; }
        let p = |i: usize| f[i].parse::<f64>().unwrap();
        match f[0] {
            "VERTEX_SE2" => { init.insert(format!("x{}", f[1]), na::dvector![p(4), p(2), p(3)]); vertex_ids.push(f[1].parse().unwrap()); }
            "EDGE_SE2" => edges.push((Edge::Se2(BetweenFactorSE2 { dx: p(3), dy: p(4), dtheta: p(5) }), vec![format!("x{}", f[1]), format!("x{}", f[2])])),
            "VERTEX_SE3:QUAT" => { is_3d = true; init.insert(format!("x{}", f[1]), na::dvector![p(5), p(6), p(7), p(8), p(2), p(3), p(4)]); vertex_ids.push(f[1].parse().unwrap()); }
            "EDGE_SE3:QUAT" => edges.push((Edge::Se3(BetweenFactorSE3 { dtx: p(3), dty: p(4), dtz: p(5), dqx: p(6), dqy: p(7), dqz: p(8), dqw: p(9) }), vec![format!("x{}", f[1]), format!("x{}", f[2])])),
            _ => {}
        }
    }
    // Wrong loop closures: identity measurements between random distant vertices.
    let n_outliers = (outlier_fraction * edges.len() as f64).round() as usize;
    let mut rng = Lcg(seed * 7919 + 13);
    for _ in 0..n_outliers {
        let (i, j) = loop {
            let i = vertex_ids[(rng.u() * vertex_ids.len() as f64) as usize];
            let j = vertex_ids[(rng.u() * vertex_ids.len() as f64) as usize];
            if i.abs_diff(j) > 10 { break (i, j); }
        };
        let e = if is_3d {
            Edge::Se3(BetweenFactorSE3 { dtx: 0.0, dty: 0.0, dtz: 0.0, dqx: 0.0, dqy: 0.0, dqz: 0.0, dqw: 1.0 })
        } else {
            Edge::Se2(BetweenFactorSE2 { dx: 0.0, dy: 0.0, dtheta: 0.0 })
        };
        edges.push((e, vec![format!("x{i}"), format!("x{j}")]));
    }
    edges.push((Edge::Prior(PriorFactor { v: init["x0"].clone() }), vec!["x0".to_string()]));

    let build = || {
        let mut problem = Problem::new();
        if is_3d {
            for k in init.keys() { problem.set_variable_manifold(k, Arc::new(SE3Manifold)); }
        }
        for (e, keys) in &edges {
            let l: Option<Box<dyn Loss + Send>> = match loss {
                "huber" => Some(Box::new(HuberLoss::new(scale))),
                "cauchy" => Some(Box::new(CauchyLoss::new(scale))),
                _ => None,
            };
            let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
            match e {
                Edge::Se2(f) => problem.add_residual_block(3, &refs, Box::new(f.clone()), l),
                Edge::Se3(f) => problem.add_residual_block(6, &refs, Box::new(f.clone()), l),
                Edge::Prior(f) => problem.add_residual_block(f.v.len(), &refs, Box::new(f.clone()), l),
            };
        }
        problem
    };
    let sum_rho = |x: &HashMap<String, na::DVector<f64>>| -> f64 {
        edges.iter().map(|(e, keys)| {
            let params: Vec<na::DVector<f64>> = keys.iter().map(|k| x[k].clone()).collect();
            let r = match e {
                Edge::Se2(f) => Factor::<f64>::residual_func(f, &params),
                Edge::Se3(f) => Factor::<f64>::residual_func(f, &params),
                Edge::Prior(f) => Factor::<f64>::residual_func(f, &params),
            };
            rho(loss, scale, r.norm_squared())
        }).sum()
    };
    let gradient_max = |problem: &Problem, x: &HashMap<String, na::DVector<f64>>| -> f64 {
        let blocks = problem.initialize_parameter_blocks(x);
        let cols = problem.get_variable_name_to_col_idx_dict(&blocks);
        let n: usize = blocks.values().map(|b| b.tangent_size()).sum();
        let sym = problem.build_symbolic_structure(&blocks, n, &cols);
        let (r, j) = problem.compute_residual_and_jacobian(&blocks, &cols, &sym);
        let g = j.as_ref().transpose() * r.as_ref();
        (0..g.nrows()).map(|i| g[(i, 0)].abs()).fold(0.0, f64::max)
    };

    // Dump the problem for the ceres harness, or evaluate a point it found.
    if let Ok(path) = std::env::var("DUMP_PROBLEM") {
        let mut out = String::new();
        let mut names: Vec<&String> = init.keys().collect();
        names.sort();
        for k in names {
            let v: Vec<String> = init[k].iter().map(|x| format!("{x:.17e}")).collect();
            out += &format!("V {} {}
", &k[1..], v.join(" "));
        }
        for (e, keys) in &edges {
            match e {
                Edge::Se2(f) => out += &format!("E2 {} {} {:.17e} {:.17e} {:.17e}
", &keys[0][1..], &keys[1][1..], f.dx, f.dy, f.dtheta),
                Edge::Se3(f) => out += &format!("E3 {} {} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}
", &keys[0][1..], &keys[1][1..], f.dtx, f.dty, f.dtz, f.dqx, f.dqy, f.dqz, f.dqw),
                Edge::Prior(f) => { let v: Vec<String> = f.v.iter().map(|x| format!("{x:.17e}")).collect(); out += &format!("P {} {}
", &keys[0][1..], v.join(" ")); }
            }
        }
        std::fs::write(path, out).unwrap();
        return;
    }
    if let Ok(path) = std::env::var("EVAL_POINT") {
        let mut x = init.clone();
        for line in read_to_string(&path).unwrap().lines() {
            let f: Vec<&str> = line.split_whitespace().collect();
            let v: Vec<f64> = f[1..].iter().map(|s| s.parse().unwrap()).collect();
            x.insert(format!("x{}", f[0]), na::DVector::from_vec(v));
        }
        let problem = build();
        let g0 = gradient_max(&problem, &init);
        println!("{file} {loss} {scale} ceres-point out={outlier_fraction} sum_rho={:.12e} grad_rel={:.3e}", sum_rho(&x), gradient_max(&problem, &x) / g0);
        return;
    }
    let options = match tol {
        "tight" => OptimizerOptions { max_iteration: 500, min_abs_error_decrease_threshold: 0.0, min_rel_error_decrease_threshold: 0.0, min_error_threshold: 0.0, ..Default::default() },
        _ => OptimizerOptions::default(),
    };
    let optimizer: Box<dyn Optimizer> = match solver {
        "gn" => Box::new(GaussNewtonOptimizer::new()),
        _ => Box::new(LevenbergMarquardtOptimizer::default()),
    };
    let problem = build();
    let g0 = gradient_max(&problem, &init);
    let mut best_time = f64::INFINITY;
    let mut result = None;
    for _ in 0..repeats {
        let problem = build();
        let start = Instant::now();
        let r = optimizer.optimize(&problem, &init, Some(options.clone()));
        best_time = best_time.min(start.elapsed().as_secs_f64());
        result = Some(r);
    }
    match result.unwrap() {
        None => println!("{file} {loss} {scale} {solver} {tol} out={outlier_fraction} edges={} sum_rho0={:.9e} FAILED", edges.len(), sum_rho(&init)),
        Some(x) => println!(
            "{file} {loss} {scale} {solver} {tol} out={outlier_fraction} edges={} sum_rho0={:.9e} sum_rho={:.12e} grad_rel={:.3e} time={:.3}",
            edges.len(), sum_rho(&init), sum_rho(&x), gradient_max(&problem, &x) / g0, best_time
        ),
    }
}
