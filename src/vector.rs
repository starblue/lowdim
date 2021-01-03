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
    Self: VectorOps<S, Self>,
    Self: for<'a> VectorOps<S, &'a Self>,
{
    /// Apply min by component
    fn min(&self, other: Self) -> Self;
    /// Apply max by component
    fn max(&self, other: Self) -> Self;

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
