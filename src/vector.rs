use core::ops;

use crate::Integer;

pub trait VectorOps<S, RHS = Self, Output = Self>
where
    Self: Sized,
    Self: ops::Add<RHS, Output = Output>,
    Self: ops::Sub<RHS, Output = Output>,
    Self: ops::Neg<Output = Output>,
{
}

pub trait Vector<S>
where
    S: Integer,
    Self: Clone + Copy,
    Self: ops::Index<usize, Output = S>,
    Self: VectorOps<S, Self>,
    Self: for<'a> VectorOps<S, &'a Self>,
{
    // Create a vector from a function which computes the coordinates.
    fn with<F>(f: F) -> Self
    where
        F: Fn(usize) -> S;

    /// Apply min by component
    fn min(&self, other: Self) -> Self {
        Self::with(|i| Integer::min(self[i], other[i]))
    }
    /// Apply max by component
    fn max(&self, other: Self) -> Self {
        Self::with(|i| Integer::max(self[i], other[i]))
    }

    /// The L1, taxicab or Manhatten norm.
    fn norm_l1(&self) -> S;

    /// Creates a vector of the vectors to orthogonal neighbours.
    ///
    /// i.e.These are the vectors with L1 norm equal to 1.
    fn unit_vecs_l1() -> Vec<Self>;

    /// The maximum, Chebychev or L∞ norm.
    fn norm_infty(&self) -> S;

    /// Creates a vector of the 8 vectors with L∞ norm equal to 1.
    fn unit_vecs_l_infty() -> Vec<Self>;
}
