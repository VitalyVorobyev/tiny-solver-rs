use std::num::NonZero;

use nalgebra as na;

use super::{AutoDiffManifold, Manifold};

/// The sphere `S^(n-1)` embedded in `R^n`, e.g. a unit direction in 3D.
///
/// Follows ceres' `SphereManifold`: the tangent space at `x` is spanned by a
/// Householder reflection that maps `x / |x|` to the last basis vector, and
/// `plus` moves along the great circle, so `|x ⊞ δ| = |x|`. The ambient
/// vector does not have to be unit length; its norm is preserved.
#[derive(Debug, Clone)]
pub struct SphereManifold {
    ambient_size: usize,
}

impl SphereManifold {
    /// A sphere in `R^ambient_size`, with tangent size `ambient_size - 1`.
    pub fn new(ambient_size: usize) -> Self {
        assert!(
            ambient_size > 1,
            "sphere ambient size must be greater than one"
        );
        Self { ambient_size }
    }

    /// Householder vector `v` and factor `beta` such that
    /// `(I - beta * v * v^T) * x = |x| * e_last`.
    fn householder<T: na::RealField>(x: na::DVectorView<T>) -> (na::DVector<T>, T) {
        let last = x.len() - 1;
        let sigma = x.rows(0, last).norm_squared();
        let pivot = x[last].clone();
        let mut v = x.into_owned();
        v[last] = T::one();

        if sigma <= T::from_f64(f64::EPSILON).unwrap() {
            // x is already (anti-)parallel to e_last.
            let beta = if pivot < T::zero() {
                T::from_f64(2.0).unwrap()
            } else {
                T::zero()
            };
            return (v, beta);
        }

        let mu = (pivot.clone() * pivot.clone() + sigma.clone()).sqrt();
        let v_pivot = if pivot <= T::zero() {
            pivot - mu
        } else {
            -sigma.clone() / (pivot + mu)
        };
        let v_pivot2 = v_pivot.clone() * v_pivot.clone();
        let beta = T::from_f64(2.0).unwrap() * v_pivot2.clone() / (sigma + v_pivot2);
        for value in v.rows_mut(0, last).iter_mut() {
            *value /= v_pivot.clone();
        }
        (v, beta)
    }

    fn apply_householder<T: na::RealField>(
        v: &na::DVector<T>,
        beta: T,
        y: na::DVector<T>,
    ) -> na::DVector<T> {
        let projection = beta * v.dot(&y);
        y - v * projection
    }
}

impl<T: na::RealField> AutoDiffManifold<T> for SphereManifold {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), self.ambient_size);
        assert_eq!(delta.len(), self.ambient_size - 1);

        // Point on the unit sphere around e_last, in the reflected frame.
        let norm2 = delta.norm_squared();
        let (cos_norm, sin_norm_over_norm) = if norm2 < T::from_f64(1e-12).unwrap() {
            // Series expansion; also keeps autodiff finite at delta = 0.
            (
                T::one() - norm2.clone() / T::from_f64(2.0).unwrap(),
                T::one() - norm2 / T::from_f64(6.0).unwrap(),
            )
        } else {
            let norm = norm2.sqrt();
            (norm.clone().cos(), norm.clone().sin() / norm)
        };
        let mut y = na::DVector::zeros(self.ambient_size);
        y.rows_mut(0, self.ambient_size - 1)
            .copy_from(&(delta * sin_norm_over_norm));
        y[self.ambient_size - 1] = cos_norm;

        let (v, beta) = Self::householder(x.clone());
        Self::apply_householder(&v, beta, y) * x.norm()
    }

    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T> {
        assert_eq!(x.len(), self.ambient_size);
        assert_eq!(y.len(), self.ambient_size);
        let last = self.ambient_size - 1;

        let (v, beta) = Self::householder(x.clone());
        let hy = Self::apply_householder(&v, beta, y.into_owned()) / x.norm();
        let head = hy.rows(0, last).into_owned();
        let head_norm2 = head.norm_squared();

        if head_norm2 <= T::from_f64(f64::EPSILON).unwrap() {
            if hy[last] < T::zero() {
                // y = -x: every direction is a shortest path, pick one.
                let mut delta = na::DVector::zeros(last);
                delta[last - 1] = T::from_f64(std::f64::consts::PI).unwrap();
                return delta;
            }
            // atan2(n, w) / n ~ 1 / w - n^2 / (3 w^3) for small n
            let w = hy[last].clone();
            let scale = T::one() / w.clone()
                - head_norm2 / (T::from_f64(3.0).unwrap() * w.clone() * w.clone() * w);
            return head * scale;
        }
        let head_norm = head_norm2.sqrt();
        let scale = head_norm.clone().atan2(hy[last].clone()) / head_norm;
        head * scale
    }
}

impl Manifold for SphereManifold {
    fn tangent_size(&self) -> NonZero<usize> {
        NonZero::new(self.ambient_size - 1).unwrap()
    }
}
