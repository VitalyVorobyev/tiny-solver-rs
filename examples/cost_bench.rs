// Micro-benchmark (not for upstream): one cost evaluation, old way vs new way.
use std::time::Instant;
use tiny_solver::helper::read_g2o;
fn main() {
    for file in ["input_M3500_g2o.g2o", "sphere2500.g2o", "parking-garage.g2o"] {
        let (problem, init) = read_g2o(&format!("tests/data/{file}"));
        let blocks = problem.initialize_parameter_blocks(&init);
        let time = |f: &dyn Fn() -> f64| {
            let mut t: Vec<f64> = (0..200).map(|_| { let s = Instant::now(); std::hint::black_box(f()); s.elapsed().as_secs_f64() * 1e3 }).collect();
            t.sort_by(f64::total_cmp);
            t[t.len() / 2]
        };
        let old = time(&|| problem.compute_residuals(&blocks, true).as_ref().squared_norm_l2());
        let new = time(&|| problem.compute_cost(&blocks));
        println!("{file:22} squared norm of corrected residuals {old:.3} ms   sum of rho {new:.3} ms   (median of 200)");
    }
}
