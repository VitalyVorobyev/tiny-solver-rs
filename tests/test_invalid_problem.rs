#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nalgebra as na;
    use tiny_solver::Optimizer;

    struct SumFactor;
    impl<T: na::RealField> tiny_solver::factors::Factor<T> for SumFactor {
        fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
            na::dvector![params[0][0].clone() + params[1][0].clone()]
        }
    }

    #[test]
    #[should_panic(expected = "variable 'y', which has no initial value")]
    fn missing_initial_value_is_reported() {
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(1, &["x", "y"], Box::new(SumFactor), None);
        let initial_values = HashMap::from([("x".to_string(), na::dvector![1.0])]);

        tiny_solver::GaussNewtonOptimizer::default().optimize(&problem, &initial_values, None);
    }

    #[test]
    #[should_panic(expected = "returned 1 residuals, but was added with dim_residual = 2")]
    fn wrong_residual_dimension_is_reported() {
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            2,
            &["x"],
            Box::new(tiny_solver::factors::PriorFactor {
                v: na::dvector![1.0],
            }),
            None,
        );
        let initial_values = HashMap::from([("x".to_string(), na::dvector![0.0])]);

        tiny_solver::GaussNewtonOptimizer::default().optimize(&problem, &initial_values, None);
    }
}
