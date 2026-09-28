#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use nalgebra as na;
    use tiny_solver::Optimizer;
    use tiny_solver::manifold::Manifold;
    use tiny_solver::manifold::sphere::SphereManifold;
    use tiny_solver::parameter_block::ParameterBlock;

    fn points() -> Vec<na::DVector<f64>> {
        vec![
            na::dvector![0.2, -0.3, 0.932],
            na::dvector![0.0, 0.0, 1.0],  // north pole
            na::dvector![0.0, 0.0, -1.0], // south pole
            na::dvector![1.0, 0.0, 0.0],
            na::dvector![-3.0, 4.0, 12.0], // not unit length
            na::dvector![0.5, -0.5, 0.5, -0.5],
        ]
    }

    fn deltas(tangent_size: usize) -> Vec<na::DVector<f64>> {
        let mut d = vec![
            na::DVector::zeros(tangent_size),
            na::DVector::from_element(tangent_size, 1e-8),
            na::DVector::from_element(tangent_size, 0.3),
        ];
        let mut v = na::DVector::zeros(tangent_size);
        v[0] = 2.5;
        d.push(v);
        d
    }

    #[test]
    fn sizes() {
        let m = SphereManifold::new(3);
        assert_eq!(m.tangent_size().get(), 2);
    }

    #[test]
    #[should_panic]
    fn rejects_ambient_size_one() {
        SphereManifold::new(1);
    }

    #[test]
    fn plus_zero_is_identity() {
        for x in points() {
            let m = SphereManifold::new(x.len());
            let zero = na::DVector::zeros(x.len() - 1);
            let y = m.plus_f64(x.as_view(), zero.as_view());
            assert!((&y - &x).norm() < 1e-12, "{x} -> {y}");
        }
    }

    #[test]
    fn plus_preserves_norm_and_minus_inverts_it() {
        for x in points() {
            let m = SphereManifold::new(x.len());
            for delta in deltas(x.len() - 1) {
                let y = m.plus_f64(x.as_view(), delta.as_view());
                assert!((y.norm() - x.norm()).abs() < 1e-12);
                let back = m.minus_f64(y.as_view(), x.as_view());
                assert!(
                    (&back - &delta).norm() < 1e-9,
                    "x={x} delta={delta} back={back}"
                );
            }
        }
    }

    #[test]
    fn plus_jacobian_spans_tangent_space() {
        // d(x ⊞ δ)/dδ at δ = 0 must have orthonormal columns (scaled by |x|)
        // that are orthogonal to x.
        for x in points() {
            let n = x.len();
            let mut block = ParameterBlock::from_vec(x.clone());
            block.set_manifold(Arc::new(SphereManifold::new(n)));
            let delta = na::DVector::from_fn(n - 1, |i, _| {
                num_dual::DualDVec64::new(
                    0.0,
                    num_dual::Derivative::some(na::DVector::from_fn(n - 1, |j, _| {
                        if i == j { 1.0 } else { 0.0 }
                    })),
                )
            });
            let y = block.plus_dual(delta.as_view());
            let jac = na::DMatrix::from_fn(n, n - 1, |r, c| {
                y[r].eps
                    .clone()
                    .unwrap_generic(na::Dyn(n - 1), na::Const::<1>)[c]
            });
            let gram = jac.transpose() * &jac / x.norm_squared();
            assert!((gram - na::DMatrix::identity(n - 1, n - 1)).norm() < 1e-9);
            assert!((jac.transpose() * &x).norm() < 1e-9);
        }
    }

    struct DirectionFactor {
        target: na::DVector<f64>,
    }
    impl<T: na::RealField> tiny_solver::factors::Factor<T> for DirectionFactor {
        fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
            params[0].clone() - self.target.clone().cast()
        }
    }

    #[test]
    fn optimize_unit_direction() {
        // Recover a unit direction; the manifold keeps the initial value's unit norm.
        let target = na::dvector![1.0, 2.0, 2.0] / 3.0;
        let mut problem = tiny_solver::Problem::new();
        problem.add_residual_block(
            3,
            &["dir"],
            Box::new(DirectionFactor {
                target: target.clone(),
            }),
            None,
        );
        problem.set_variable_manifold("dir", Arc::new(SphereManifold::new(3)));
        let initial_values = HashMap::from([("dir".to_string(), na::dvector![0.36, 0.48, 0.8])]);

        let result = tiny_solver::LevenbergMarquardtOptimizer::default()
            .optimize(&problem, &initial_values, None)
            .unwrap();

        let dir = &result["dir"];
        assert!((dir.norm() - 1.0).abs() < 1e-12);
        assert!(
            (dir - &target).norm() < 1e-6,
            "dir = {:?}, expected {:?}",
            dir.as_slice(),
            target.normalize().as_slice()
        );
    }
}
