// NIST StRD nonlinear regression with tiny-solver's LM. A port of Ceres 2.2
// `examples/nist.cc`: same parser, same 27 models, same LRE and success rule (LRE > 4).
// See eval/nist/README.md. From the repo root:
//   RAYON_NUM_THREADS=1 cargo run --release --example nist -- eval/nist/data
// Columns: problem start options lre status none accepted rejected invalid cost certified_rss ms
// stop (the last stopping message; max_iter if none).
// Counts come from LM's log messages: "iter:" accepted, "u " rejected, "solve ax=b failed" invalid.
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nalgebra as na;
use tiny_solver::{LevenbergMarquardtOptimizer, Optimizer, OptimizerOptions, Problem};

static ACCEPTED: AtomicUsize = AtomicUsize::new(0);
static REJECTED: AtomicUsize = AtomicUsize::new(0);
static INVALID: AtomicUsize = AtomicUsize::new(0);
// The last stopping message: 1 abs decrease, 2 rel decrease, 3 error too low, 4 step size.
static STOP: AtomicUsize = AtomicUsize::new(0);

struct Counter;
impl log::Log for Counter {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        let m = record.args().to_string();
        if m.starts_with("iter:") {
            ACCEPTED.fetch_add(1, Ordering::Relaxed);
        } else if m.starts_with("u ") {
            REJECTED.fetch_add(1, Ordering::Relaxed);
        } else if m.starts_with("solve ax=b failed") {
            INVALID.fetch_add(1, Ordering::Relaxed);
        } else if m.starts_with("absolute error decrease low") {
            STOP.store(1, Ordering::Relaxed);
        } else if m.starts_with("relative error decrease low") {
            STOP.store(2, Ordering::Relaxed);
        } else if m.starts_with("error too low") {
            STOP.store(3, Ordering::Relaxed);
        } else if m.starts_with("relative step size low") {
            STOP.store(4, Ordering::Relaxed);
        }
    }
    fn flush(&self) {}
}

#[derive(Clone, Copy)]
enum Model {
    Bennett5,
    BoxBOD,
    Chwirut,
    DanWood,
    Gauss,
    Lanczos,
    Hahn1,
    Kirby2,
    MGH09,
    MGH10,
    MGH17,
    Misra1a,
    Misra1b,
    Misra1c,
    Misra1d,
    Roszman1,
    Rat42,
    Rat43,
    Thurber,
    Enso,
    Eckerle4,
    Nelson,
}

fn c<T: na::RealField>(v: f64) -> T {
    T::from_f64(v).unwrap()
}

/// The model prediction at one observation; `x` holds its predictors.
fn predict<T: na::RealField>(m: Model, b: &[T], x: &[f64]) -> T {
    let xt: T = c(x[0]);
    let one = T::one();
    let pi = c::<T>(std::f64::consts::PI);
    let sq = |v: T| v.clone() * v;
    match m {
        Model::Bennett5 => b[0].clone() * (b[1].clone() + xt).powf(-one / b[2].clone()),
        Model::BoxBOD | Model::Misra1a => b[0].clone() * (one - (-b[1].clone() * xt).exp()),
        Model::Chwirut => (-b[0].clone() * xt.clone()).exp() / (b[1].clone() + b[2].clone() * xt),
        Model::DanWood => b[0].clone() * xt.powf(b[1].clone()),
        Model::Gauss => {
            b[0].clone() * (-b[1].clone() * xt.clone()).exp()
                + b[2].clone() * (-sq((xt.clone() - b[3].clone()) / b[4].clone())).exp()
                + b[5].clone() * (-sq((xt - b[6].clone()) / b[7].clone())).exp()
        }
        Model::Lanczos => {
            b[0].clone() * (-b[1].clone() * xt.clone()).exp()
                + b[2].clone() * (-b[3].clone() * xt.clone()).exp()
                + b[4].clone() * (-b[5].clone() * xt).exp()
        }
        Model::Hahn1 | Model::Thurber => {
            let (x2, x3) = (sq(xt.clone()), sq(xt.clone()) * xt.clone());
            (b[0].clone()
                + b[1].clone() * xt.clone()
                + b[2].clone() * x2.clone()
                + b[3].clone() * x3.clone())
                / (one + b[4].clone() * xt + b[5].clone() * x2 + b[6].clone() * x3)
        }
        Model::Kirby2 => {
            let x2 = sq(xt.clone());
            (b[0].clone() + b[1].clone() * xt.clone() + b[2].clone() * x2.clone())
                / (one + b[3].clone() * xt + b[4].clone() * x2)
        }
        Model::MGH09 => {
            let x2 = sq(xt.clone());
            b[0].clone() * (x2.clone() + xt.clone() * b[1].clone())
                / (x2 + xt * b[2].clone() + b[3].clone())
        }
        Model::MGH10 => b[0].clone() * (b[1].clone() / (xt + b[2].clone())).exp(),
        Model::MGH17 => {
            b[0].clone()
                + b[1].clone() * (-xt.clone() * b[3].clone()).exp()
                + b[2].clone() * (-xt * b[4].clone()).exp()
        }
        Model::Misra1b => {
            let d = one.clone() + b[1].clone() * xt / c(2.0);
            b[0].clone() * (one - one_over(sq(d)))
        }
        Model::Misra1c => {
            b[0].clone() * (one.clone() - (one + c::<T>(2.0) * b[1].clone() * xt).powf(c(-0.5)))
        }
        Model::Misra1d => b[0].clone() * b[1].clone() * xt.clone() / (one + b[1].clone() * xt),
        Model::Roszman1 => {
            b[0].clone() - b[1].clone() * xt.clone() - b[2].clone().atan2(xt - b[3].clone()) / pi
        }
        Model::Rat42 => b[0].clone() / (one + (b[1].clone() - b[2].clone() * xt).exp()),
        Model::Rat43 => {
            b[0].clone()
                / (one.clone() + (b[1].clone() - b[2].clone() * xt).exp()).powf(one / b[3].clone())
        }
        Model::Enso => {
            let w = c::<T>(2.0) * pi * xt;
            let twelve = c::<T>(12.0);
            b[0].clone()
                + b[1].clone() * (w.clone() / twelve.clone()).cos()
                + b[2].clone() * (w.clone() / twelve).sin()
                + b[4].clone() * (w.clone() / b[3].clone()).cos()
                + b[5].clone() * (w.clone() / b[3].clone()).sin()
                + b[7].clone() * (w.clone() / b[6].clone()).cos()
                + b[8].clone() * (w / b[6].clone()).sin()
        }
        Model::Eckerle4 => {
            b[0].clone() / b[1].clone()
                * (c::<T>(-0.5) * sq((xt - b[2].clone()) / b[1].clone())).exp()
        }
        Model::Nelson => b[0].clone() - b[1].clone() * xt * (-b[2].clone() * c::<T>(x[1])).exp(),
    }
}

fn one_over<T: na::RealField>(v: T) -> T {
    T::one() / v
}

struct Fit {
    model: Model,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
}

impl<T: na::RealField> tiny_solver::factors::Factor<T> for Fit {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        let b: Vec<T> = params[0].iter().cloned().collect();
        na::DVector::from_iterator(
            self.y.len(),
            self.x.iter().zip(&self.y).map(|(x, &y)| {
                // Nelson fits log(y), as in nist.cc.
                let y = if matches!(self.model, Model::Nelson) {
                    y.ln()
                } else {
                    y
                };
                c::<T>(y) - predict(self.model, &b, x)
            }),
        )
    }
}

struct Nist {
    starts: Vec<Vec<f64>>,
    certified: Vec<f64>,
    certified_rss: f64,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
}

/// The same line-based parser as nist.cc's NISTProblem.
fn read_nist(path: &str) -> Nist {
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<Vec<&str>> = text
        .lines()
        .map(|l| {
            l.trim_end_matches('\r')
                .split(' ')
                .filter(|s| !s.is_empty())
                .collect()
        })
        .collect();
    let num_responses: usize = lines[24][1].parse().unwrap();
    let num_predictors: usize = lines[25][0].parse().unwrap();
    let num_observations: usize = lines[26][0].parse().unwrap();
    let num_parameters: usize = lines[31][0].parse().unwrap();
    let num_tries = lines[40].len() - 4;
    let mut starts = vec![vec![0.0; num_parameters]; num_tries];
    let mut certified = vec![0.0; num_parameters];
    for p in 0..num_parameters {
        let l = &lines[40 + p];
        for (t, start) in starts.iter_mut().enumerate() {
            start[p] = l[t + 2].parse().unwrap();
        }
        certified[p] = l[2 + num_tries].parse().unwrap();
    }
    let rss_line = 40 + num_parameters + 1;
    let certified_rss = lines[rss_line][4].parse().unwrap();
    let first = rss_line + 1 + 18 - num_parameters;
    let (mut x, mut y) = (vec![], vec![]);
    for l in &lines[first..first + num_observations] {
        y.push(l[0].parse().unwrap());
        x.push(
            (0..num_predictors)
                .map(|j| l[j + num_responses].parse().unwrap())
                .collect(),
        );
    }
    Nist {
        starts,
        certified,
        certified_rss,
        x,
        y,
    }
}

fn lre(expected: &[f64], actual: &[f64]) -> f64 {
    expected
        .iter()
        .zip(actual)
        .map(|(e, a)| (-((e - a).abs() / e.abs()).log10()).clamp(0.0, 11.0))
        .fold(12.0, f64::min)
}

fn main() {
    log::set_logger(&Counter).unwrap();
    let dir = std::env::args().nth(1).expect("data dir");
    let problems = [
        ("Misra1a", Model::Misra1a),
        ("Chwirut1", Model::Chwirut),
        ("Chwirut2", Model::Chwirut),
        ("Lanczos3", Model::Lanczos),
        ("Gauss1", Model::Gauss),
        ("Gauss2", Model::Gauss),
        ("DanWood", Model::DanWood),
        ("Misra1b", Model::Misra1b),
        ("Kirby2", Model::Kirby2),
        ("Hahn1", Model::Hahn1),
        ("Nelson", Model::Nelson),
        ("MGH17", Model::MGH17),
        ("Lanczos1", Model::Lanczos),
        ("Lanczos2", Model::Lanczos),
        ("Gauss3", Model::Gauss),
        ("Misra1c", Model::Misra1c),
        ("Misra1d", Model::Misra1d),
        ("Roszman1", Model::Roszman1),
        ("ENSO", Model::Enso),
        ("MGH09", Model::MGH09),
        ("Thurber", Model::Thurber),
        ("BoxBOD", Model::BoxBOD),
        ("Rat42", Model::Rat42),
        ("MGH10", Model::MGH10),
        ("Eckerle4", Model::Eckerle4),
        ("Rat43", Model::Rat43),
        ("Bennett5", Model::Bennett5),
    ];
    for opts in ["tight", "default"] {
        let options = if opts == "tight" {
            OptimizerOptions {
                max_iteration: 10000,
                min_abs_error_decrease_threshold: 0.0,
                min_rel_error_decrease_threshold: 0.0,
                min_error_threshold: 0.0,
                ..Default::default()
            }
        } else {
            OptimizerOptions::default()
        };
        let mut solved = 0;
        for (name, model) in problems {
            let nist = read_nist(&format!("{dir}/{name}.dat"));
            let mut problem = Problem::new();
            let fit = Fit {
                model,
                x: nist.x.clone(),
                y: nist.y.clone(),
            };
            problem.add_residual_block(nist.y.len(), &["b"], Box::new(fit), None);
            for (s, start) in nist.starts.iter().enumerate() {
                let init = HashMap::from([("b".to_string(), na::DVector::from_vec(start.clone()))]);
                let lm = LevenbergMarquardtOptimizer::default();
                for c in [&ACCEPTED, &REJECTED, &INVALID, &STOP] {
                    c.store(0, Ordering::Relaxed);
                }
                log::set_max_level(log::LevelFilter::Trace);
                let t = Instant::now();
                let result = lm.optimize(&problem, &init, Some(options.clone()));
                let ms = t.elapsed().as_secs_f64() * 1e3;
                log::set_max_level(log::LevelFilter::Off);
                let (a, r, i) = (
                    ACCEPTED.load(Ordering::Relaxed),
                    REJECTED.load(Ordering::Relaxed),
                    INVALID.load(Ordering::Relaxed),
                );
                let (l, cost, none) = match &result {
                    Some(x) => {
                        let b: Vec<f64> = x["b"].iter().copied().collect();
                        let cost = problem.compute_cost(&problem.initialize_parameter_blocks(x));
                        (lre(&nist.certified, &b), cost, false)
                    }
                    None => (0.0, f64::NAN, true),
                };
                let stop = [
                    "max_iter",
                    "abs_decrease",
                    "rel_decrease",
                    "error_low",
                    "step_size",
                ][STOP.load(Ordering::Relaxed)];
                let ok = l > 4.0;
                solved += ok as usize;
                println!(
                    "{name} {} {opts} {l:.1} {} {} {a} {r} {i} {cost:.10e} {:.10e} {ms:.1} {stop}",
                    s + 1,
                    if ok { "SUCCESS" } else { "FAILURE" },
                    if none { "none" } else { "some" },
                    nist.certified_rss
                );
            }
        }
        println!("# {opts}: solved {solved}/54");
    }
}
