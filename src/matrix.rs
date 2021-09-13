//! Traits for matrices.

use core::ops;

use crate::Integer;
use crate::Vector;

/// Required arithmetic operations for matrices.
///
/// Must be in a separate trait to allow `Self` to be a reference type
/// and the output the base type.
pub trait MatrixOps<S, RHS = Self, Output = Self>
where
    Self: Sized,
    Self: ops::Mul<RHS, Output = Output>,
{
}

/// Required traits and operations for matrices.
pub trait Matrix<S>
where
    S: Integer,
    Self: Clone,
    Self: MatrixOps<S, Self>,
    Self: for<'a> MatrixOps<S, &'a Self>,
    Self: MatrixOps<S, <Self as Matrix<S>>::V, <Self as Matrix<S>>::V>,
    Self: for<'a> MatrixOps<S, &'a <Self as Matrix<S>>::V, <Self as Matrix<S>>::V>,
{
    /// The corresponding vector type.
    ///
    /// Must have the same scalar base type and dimension.
    type V: Vector<S>;

    /// Creates a new matrix from a function.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::convert::TryFrom;
    /// # use lowdim::Matrix2d;
    /// let m = Matrix2d::with(|i, j| i64::try_from(3 * i + j).unwrap());
    /// assert_eq!(Matrix2d::new(0, 1, 3, 4), m);
    /// ```
    fn with<F>(f: F) -> Self
    where
        F: Fn(usize, usize) -> S;

    /// Creates a zero matrix.
    fn zero() -> Self {
        Self::with(|_i, _j| S::zero())
    }

    /// Creates a unit matrix.
    fn unit() -> Self {
        Self::with(|i, j| if i == j { S::one() } else { S::zero() })
    }
}
