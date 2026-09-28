#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use nalgebra as na;
    use tiny_solver::Optimizer;
    use tiny_solver::factors::BetweenFactorSE3;
    use tiny_solver::manifold::se3::SE3Manifold;

    fn measurement() -> (na::UnitQuaternion<f64>, BetweenFactorSE3) {
        let q = na::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let factor = BetweenFactorSE3 {
            dtx: 1.0,
            dty: 0.0,
            dtz: 0.0,
            dqx: q.i,
            dqy: q.j,
            dqz: q.k,
            dqw: q.w,
        };
        (q, factor)
    }

    /// Two SE3 poses joined by one between factor. x0 has the SE3 manifold and
    /// all seven of its coordinates fixed, which should anchor the gauge.
    fn anchored_pose_graph() -> (tiny_solver::Problem, HashMap<String, na::DVector<f64>>) {
        let (_, factor) = measurement();
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(6, &["x0", "x1"], Box::new(factor), None);
        problem.set_variable_manifold("x0", Arc::new(SE3Manifold));
        problem.set_variable_manifold("x1", Arc::new(SE3Manifold));
        for idx in 0..7 {
            problem.fix_variable("x0", idx);
        }
        let initial_values = HashMap::from([
            (
                "x0".to_string(),
                na::dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
            (
                "x1".to_string(),
                na::dvector![0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0],
            ),
        ]);
        (problem, initial_values)
    }

    #[test]
    fn fully_fixed_manifold_block_has_no_jacobian_columns() {
        let (problem, initial_values) = anchored_pose_graph();
        let parameter_blocks = problem.initialize_parameter_blocks(&initial_values);
        let variable_name_to_col_idx_dict =
            problem.get_variable_name_to_col_idx_dict(&parameter_blocks);
        let symbolic_structure =
            problem.build_symbolic_structure(&parameter_blocks, 6, &variable_name_to_col_idx_dict);

        let (_, jac) = problem.compute_residual_and_jacobian(
            &parameter_blocks,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        );

        assert_eq!(jac.ncols(), 6);
        assert!(parameter_blocks["x0"].is_constant());
        assert_eq!(parameter_blocks["x0"].effective_tangent_size(), 0);
        assert!(!parameter_blocks["x1"].is_constant());
        assert_eq!(parameter_blocks["x1"].effective_tangent_size(), 6);
    }

    fn assert_solution(result: &HashMap<String, na::DVector<f64>>) {
        let (q, _) = measurement();
        assert_eq!(
            result["x0"],
            na::dvector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        );
        let x1 = &result["x1"];
        // x0 is the identity, so x1 must equal the measurement (quaternion up to sign).
        let sign = if x1[3] * q.w < 0.0 { -1.0 } else { 1.0 };
        let expected = na::dvector![q.i, q.j, q.k, q.w, 1.0, 0.0, 0.0];
        for i in 0..4 {
            assert!((sign * x1[i] - expected[i]).abs() < 1e-6, "x1 = {x1}");
        }
        for i in 4..7 {
            assert!((x1[i] - expected[i]).abs() < 1e-6, "x1 = {x1}");
        }
    }

    #[test]
    fn gauss_newton_holds_fully_fixed_manifold_block_constant() {
        let (problem, initial_values) = anchored_pose_graph();
        let result = tiny_solver::GaussNewtonOptimizer::default()
            .optimize(&problem, &initial_values, None)
            .expect("the gauge is anchored, so the system is not singular");
        assert_solution(&result);
    }

    #[test]
    fn levenberg_marquardt_holds_fully_fixed_manifold_block_constant() {
        let (problem, initial_values) = anchored_pose_graph();
        let result = tiny_solver::LevenbergMarquardtOptimizer::default()
            .optimize(&problem, &initial_values, None)
            .unwrap();
        assert_solution(&result);
    }
}
