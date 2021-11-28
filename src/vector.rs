//! Traits for vectors.

use core::cmp::Ordering;
use core::iter;
use core::iter::once;
use core::ops;

use crate::Integer;

/// Required arithmetic operations for vectors.
///
/// Must be in a separate trait to allow `Self` to be a reference type
/// and the output the base type.
pub trait VectorOps<S, RHS = Self, Output = Self>
where
    Self: Sized,
    Self: ops::Add<RHS, Output = Output>,
    Self: ops::Sub<RHS, Output = Output>,
    Self: ops::Neg<Output = Output>,
    Self: ops::Mul<RHS, Output = S>,
{
}

/// Required traits and operations for vectors.
pub trait Vector<S>
where
    S: Integer,
    Self: Clone + Copy,
    Self: ops::Index<usize, Output = S>,
    Self: PartialOrd,
    Self: iter::FromIterator<S>,
    Self: VectorOps<S, Self>,
    Self: for<'a> VectorOps<S, &'a Self>,
    Self: ops::Div<S, Output = Self>,
    Self: for<'a> ops::Div<&'a S, Output = Self>,
{
    /// The dimension of the vectors in this type.
    const DIM: usize;

    /// Create a vector from a function which computes the coordinates.
    ///
    /// The function must return a scalar value for each possible coordinate index.
    ///
    /// # Example
    /// ```
    /// # use std::convert::TryFrom;
    /// # use lowdim::v4d;
    /// # use lowdim::Vec4d;
    /// # use lowdim::Vector;
    /// assert_eq!(v4d(0, 1, 2, 3), Vec4d::with(|i| i64::try_from(i).unwrap()));
    /// ```
    fn with<F>(f: F) -> Self
    where
        F: Fn(usize) -> S;

    /// Returns a slice containing the coordinates of the vector.
    fn as_slice(&self) -> &[S];

    /// Returns a mutable slice containing the coordinates of the vector.
    fn as_mut_slice(&mut self) -> &mut [S];

    /// Creates the zero vector.
    fn zero() -> Self {
        Self::with(|_| S::zero())
    }
    /// Creates a vector of ones.
    fn ones() -> Self {
        Self::with(|_| S::one())
    }

    /// Returns `true` if a vector is the zero vector.
    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }

    /// Apply min by component
    fn min(&self, other: Self) -> Self {
        Self::with(|i| self[i].min(other[i]))
    }
    /// Apply max by component
    fn max(&self, other: Self) -> Self {
        Self::with(|i| self[i].max(other[i]))
    }

    /// Signum by component.
    ///
    /// Maps a vector to a unit step in the L∞ norm.
    /// This is a step on a shortest path w.r.t. L∞ along the vector.
    fn signum(&self) -> Self {
        Self::with(|i| self[i].signum())
    }

    /// Returns the L1 norm of the vector.
    ///
    /// This is also called the taxicab, Manhatten or city block norm.
    fn norm_l1(&self) -> S;

    /// Creates a vector of the unit vectors.
    fn unit_vecs() -> Vec<Self> {
        (0..Self::DIM)
            .map(|i| Self::with(|j| S::from(if i == j { 1 } else { 0 })))
            .collect::<Vec<_>>()
    }

    /// Creates a vector of the vectors to orthogonal neighbours.
    ///
    /// These are the vectors with L1 norm equal to 1.
    fn unit_vecs_l1() -> Vec<Self> {
        Self::unit_vecs()
            .into_iter()
            .flat_map(|uv| once(uv).chain(once(-uv)))
            .collect::<Vec<_>>()
    }

    /// Returns the L∞ norm of the vector.
    ///
    /// This is also called the maximum or Chebychev norm.
    fn norm_l_infty(&self) -> S;

    /// Returns the square of the L2-norm of the vector.
    ///
    /// The L2-norm is also called the Euclidean norm and
    /// is the standard notion of the length of a vector.
    fn norm_l2_squared(&self) -> S;

    /// Creates a vector of the vectors with L∞ norm equal to 1.
    ///
    /// These correspond to a single orthogonal or diagonal step.
    fn unit_vecs_l_infty() -> Vec<Self>;

    /// Returns the lexicographic total ordering for this and another vector.
    ///
    /// That is, the first different coordinate decides the ordering.
    /// This is useful as an arbitrary total ordering for sorting,
    /// but is not intended to be otherwise meaningful.
    fn lex_cmp(&self, other: &Self) -> Ordering;
}

#[doc(hidden)]
#[macro_export]
macro_rules! scalar_mul {
    ($s:ty, $v:ty) => {
        impl std::ops::Mul<$v> for $s {
            type Output = $v;

            fn mul(self, other: $v) -> $v {
                <$v>::with(|i| self * other[i])
            }
        }
        impl<'a> std::ops::Mul<&'a $v> for $s {
            type Output = $v;

            fn mul(self, other: &'a $v) -> $v {
                <$v>::with(|i| self * other[i])
            }
        }
        impl<'a> std::ops::Mul<$v> for &'a $s {
            type Output = $v;

            fn mul(self, other: $v) -> $v {
                <$v>::with(|i| self * other[i])
            }
        }
        impl<'a> std::ops::Mul<&'a $v> for &'a $s {
            type Output = $v;

            fn mul(self, other: &'a $v) -> $v {
                <$v>::with(|i| self * other[i])
            }
        }
    };
}
