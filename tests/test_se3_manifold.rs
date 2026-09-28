#[cfg(test)]
mod tests {
    use nalgebra as na;
    use tiny_solver::manifold::Manifold;
    use tiny_solver::manifold::se3::{SE3, SE3Manifold};
    use tiny_solver::manifold::so3::QuaternionManifold;

    /// A few poses as [qx, qy, qz, qw, tx, ty, tz].
    fn poses() -> Vec<na::Isometry3<f64>> {
        vec![
            na::Isometry3::identity(),
            na::Isometry3::new(
                na::Vector3::new(1.0, -2.0, 0.5),
                na::Vector3::new(0.1, 0.2, 0.3),
            ),
            na::Isometry3::new(
                na::Vector3::new(-0.3, 0.0, 4.0),
                na::Vector3::new(2.0, -1.0, 0.5),
            ),
        ]
    }

    fn to_vec(pose: &na::Isometry3<f64>) -> na::DVector<f64> {
        let q = pose.rotation;
        let t = pose.translation.vector;
        na::dvector![q.i, q.j, q.k, q.w, t.x, t.y, t.z]
    }

    fn from_vec(v: &na::DVector<f64>) -> na::Isometry3<f64> {
        let q = na::UnitQuaternion::from_quaternion(na::Quaternion::new(v[3], v[0], v[1], v[2]));
        na::Isometry3::from_parts(na::Translation3::new(v[4], v[5], v[6]), q)
    }

    fn deltas() -> Vec<na::DVector<f64>> {
        vec![
            na::DVector::zeros(6),
            na::dvector![1e-9, -2e-9, 3e-9, 1e-9, 0.0, -1e-9],
            na::dvector![0.1, -0.2, 0.3, 1.0, 2.0, -3.0],
            na::dvector![1.5, 0.5, -1.0, -0.5, 0.0, 0.25],
        ]
    }

    fn assert_pose_eq(a: &na::Isometry3<f64>, b: &na::Isometry3<f64>) {
        assert!(a.rotation.angle_to(&b.rotation) < 1e-9, "{a} != {b}");
        assert!(
            (a.translation.vector - b.translation.vector).norm() < 1e-9,
            "{a} != {b}"
        );
    }

    #[test]
    fn se3_compose_inverse_and_transform_match_nalgebra() {
        let point = na::Vector3::new(0.3, -1.2, 2.0);
        for a in poses() {
            for b in poses() {
                let se3_a = SE3::from_vec(to_vec(&a).as_view());
                let se3_b = SE3::from_vec(to_vec(&b).as_view());
                let composed = from_vec(&(&se3_a * &se3_b).to_dvec());
                assert_pose_eq(&composed, &(a * b));
            }
            let se3_a = SE3::from_vec(to_vec(&a).as_view());
            assert_pose_eq(&from_vec(&se3_a.inverse().to_dvec()), &a.inverse());
            let transformed = &se3_a * point.as_view();
            assert!((transformed - a.transform_point(&point.into()).coords).norm() < 1e-12);
        }
    }

    /// x ⊞ δ = x * (exp(δ[0..3]), δ[3..6]): the rotation is perturbed on the
    /// right and the translation moves by R * δt (SO(3) x R^3, not the SE(3)
    /// exponential).
    #[test]
    fn se3_plus_is_right_perturbation() {
        let m = SE3Manifold;
        for pose in poses() {
            for delta in deltas() {
                let x = to_vec(&pose);
                let y = from_vec(&m.plus_f64(x.as_view(), delta.as_view()));
                let rotation = pose.rotation
                    * na::UnitQuaternion::from_scaled_axis(na::Vector3::new(
                        delta[0], delta[1], delta[2],
                    ));
                let translation = pose.translation.vector
                    + pose.rotation * na::Vector3::new(delta[3], delta[4], delta[5]);
                assert_pose_eq(&y, &na::Isometry3::from_parts(translation.into(), rotation));
            }
        }
    }

    #[test]
    fn plus_zero_is_identity_and_minus_inverts_plus() {
        let se3 = SE3Manifold;
        let quaternion = QuaternionManifold;
        for pose in poses() {
            let x = to_vec(&pose);
            let q = x.rows(0, 4).into_owned();
            for delta in deltas() {
                let y = se3.plus_f64(x.as_view(), delta.as_view());
                if delta.norm() == 0.0 {
                    assert!((&y - &x).norm() < 1e-15);
                }
                assert!((y.rows(0, 4).norm() - 1.0).abs() < 1e-12);
                let back = se3.minus_f64(y.as_view(), x.as_view());
                assert!((&back - &delta).norm() < 1e-9, "{back} != {delta}");

                let d_rot = delta.rows(0, 3).into_owned();
                let y = quaternion.plus_f64(q.as_view(), d_rot.as_view());
                assert!((y.norm() - 1.0).abs() < 1e-12);
                let back = quaternion.minus_f64(y.as_view(), q.as_view());
                assert!((&back - &d_rot).norm() < 1e-9, "{back} != {d_rot}");
            }
        }
    }

    /// The solver differentiates `plus` at δ = 0 with dual numbers; that
    /// Jacobian must match a central finite difference.
    #[test]
    fn plus_jacobian_at_zero_matches_finite_differences() {
        let m = SE3Manifold;
        let h = 1e-6;
        for pose in poses() {
            let x = to_vec(&pose);
            let x_dual = x.map(num_dual::DualDVec64::from_re);
            let delta = na::DVector::from_fn(6, |i, _| {
                num_dual::DualDVec64::new(
                    0.0,
                    num_dual::Derivative::some(na::DVector::from_fn(6, |j, _| {
                        if i == j { 1.0 } else { 0.0 }
                    })),
                )
            });
            let y = m.plus_dual(x_dual.as_view(), delta.as_view());
            for c in 0..6 {
                let mut step = na::DVector::zeros(6);
                step[c] = h;
                let forward = m.plus_f64(x.as_view(), step.as_view());
                let backward = m.plus_f64(x.as_view(), (-&step).as_view());
                let numeric = (forward - backward) / (2.0 * h);
                for r in 0..7 {
                    let dual = y[r].eps.clone().unwrap_generic(na::Dyn(6), na::Const::<1>)[c];
                    assert!((dual - numeric[r]).abs() < 1e-8, "d y[{r}] / d delta[{c}]");
                }
            }
        }
    }
}
