//! Traits for vectors.

#![warn(missing_docs)]

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
{
}

/// Required traits and operations for vectors.
pub trait Vector<S>
where
    S: Integer,
    Self: Clone + Copy,
    Self: ops::Index<usize, Output = S>,
    Self: PartialOrd,
    Self: VectorOps<S, Self>,
    Self: for<'a> VectorOps<S, &'a Self>,
{
    /// Create a vector from a function which computes the coordinates.
    ///
    /// The function must return a scalar value for each possible coordinate index.
    ///
    /// # Example
    /// ```
    /// # use std::convert::TryFrom;
    /// # use gamedim::v4d;
    /// # use gamedim::Vec4d;
    /// # use gamedim::Vector;
    /// assert_eq!(v4d(0, 1, 2, 3), Vec4d::with(|i| i64::try_from(i).unwrap()));
    /// ```
    fn with<F>(f: F) -> Self
    where
        F: Fn(usize) -> S;

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

    /// The L1, taxicab or Manhatten norm.
    fn norm_l1(&self) -> S;

    /// Creates a vector of the unit vectors.
    fn unit_vecs() -> Vec<Self>;

    /// Creates a vector of the vectors to orthogonal neighbours.
    ///
    /// These are the vectors with L1 norm equal to 1.
    fn unit_vecs_l1() -> Vec<Self>;

    /// The maximum, Chebychev or L∞ norm.
    fn norm_l_infty(&self) -> S;

    /// Creates a vector of the 8 vectors with L∞ norm equal to 1.
    fn unit_vecs_l_infty() -> Vec<Self>;
}
