#[cfg(test)]
mod tests {
    use tiny_solver::Optimizer;
    use tiny_solver::helper::read_g2o;
    use tiny_solver::optimizer::OptimizerOptions;

    /// Solving the same problem twice must give bit-identical results, even
    /// though every HashMap gets its own random iteration order.
    #[test]
    fn repeated_solves_are_bitwise_identical() {
        let solve = || {
            let (problem, initial_values) = read_g2o("tests/data/input_M3500_g2o.g2o");
            let options = OptimizerOptions {
                max_iteration: 2,
                ..Default::default()
            };
            tiny_solver::GaussNewtonOptimizer::default()
                .optimize(&problem, &initial_values, Some(options))
                .unwrap()
        };

        let reference = solve();
        for _ in 0..2 {
            let result = solve();
            for (key, value) in &reference {
                let bits = |v: &tiny_solver::na::DVector<f64>| -> Vec<u64> {
                    v.iter().map(|x| x.to_bits()).collect()
                };
                assert_eq!(bits(&result[key]), bits(value), "variable {key}");
            }
        }
    }
}
