#[cfg(test)]
mod tests {
    use core::f64;
    use tiny_solver::loss_functions::*;

    #[test]
    fn arctan_loss() {
        let tolerance = 100.0;
        let asymptote = tolerance * f64::consts::PI / 2.0;

        let arctan_loss = ArctanLoss::new(tolerance);

        let rho1 = arctan_loss.evaluate(1.0);
        let rho2 = arctan_loss.evaluate(30.0);

        // Test that rho[0] grows linearly with the scale
        assert!(rho1[0] < rho2[0]);

        // Test that rho[1] decreases linearly with the scale
        assert!(rho1[1] > rho2[1]);

        // Test that scales largely above the tolerance are asymptotically bounded
        assert!(arctan_loss.evaluate(tolerance * tolerance * tolerance)[0] < asymptote);
    }

    fn assert_close(actual: [f64; 3], expected: [f64; 3]) {
        for i in 0..3 {
            assert!(
                (actual[i] - expected[i]).abs() < 1e-12,
                "rho[{i}]: {} != {}",
                actual[i],
                expected[i]
            );
        }
    }

    #[test]
    fn huber_loss_values() {
        let loss = HuberLoss::new(1.5);
        // inlier: s <= scale^2
        assert_close(loss.evaluate(1.0), [1.0, 1.0, 0.0]);
        // outlier: rho = 2 a sqrt(s) - a^2, rho' = a / sqrt(s), rho'' = -rho' / (2 s)
        assert_close(loss.evaluate(4.0), [3.75, 0.75, -0.09375]);
    }

    #[test]
    fn cauchy_loss_values() {
        let loss = CauchyLoss::new(2.0);
        // rho = a^2 ln(1 + s / a^2), rho' = 1 / (1 + s / a^2), rho'' = -1 / (a^2 (1 + s / a^2)^2)
        let s = 8.0;
        assert_close(
            loss.evaluate(s),
            [4.0 * 3.0f64.ln(), 1.0 / 3.0, -1.0 / 36.0],
        );
    }

    #[test]
    fn arctan_loss_values() {
        let loss = ArctanLoss::new(1.0);
        // rho = a atan(s / a), rho' = 1 / (1 + s^2 / a^2), rho'' = -2 s / a^2 / (1 + s^2 / a^2)^2
        assert_close(loss.evaluate(1.0), [f64::consts::FRAC_PI_4, 0.5, -0.5]);
    }

    /// Like ceres, rho' is clamped to the smallest positive double, so it never
    /// reaches zero even when s overflows the internal terms.
    #[test]
    fn first_derivative_stays_positive() {
        let s = f64::INFINITY;
        assert!(HuberLoss::new(1.0).evaluate(s)[1] > 0.0);
        assert!(CauchyLoss::new(1.0).evaluate(s)[1] > 0.0);
        assert!(ArctanLoss::new(1.0).evaluate(s)[1] > 0.0);
    }

    #[test]
    #[should_panic]
    fn cauchy_loss_rejects_non_positive_scale() {
        CauchyLoss::new(0.0);
    }
}
