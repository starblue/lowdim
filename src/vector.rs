//! Traits for vectors.

use core::cmp::Ordering;
use core::hash::Hash;
use core::iter;
use core::marker::PhantomData;
use core::ops;

use crate::Integer;
use crate::Layout;

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
    Self: Clone + Copy + Eq + Hash,
    Self: ops::Index<usize, Output = S>,
    Self: iter::FromIterator<S>,
    Self: VectorOps<S, Self>,
    Self: for<'a> VectorOps<S, &'a Self>,
    Self: ops::Div<S, Output = Self>,
    Self: for<'a> ops::Div<&'a S, Output = Self>,
    Self: iter::Sum<Self> + for<'a> iter::Sum<&'a Self>,
{
    /// The dimension of the vectors in this type.
    const DIM: usize;

    /// The default layout to use with this vector.
    type DefaultLayout: Layout<S, Self>;

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

    /// Returns an iterator that yields the unit vectors.
    fn unit_vecs() -> UnitVecs<S, Self> {
        UnitVecs::new()
    }

    /// Returns an iterator that yields the vectors to orthogonal neighbours.
    ///
    /// These are the vectors with L1 norm equal to 1.
    fn unit_vecs_l1() -> UnitVecsL1<S, Self> {
        UnitVecsL1::new()
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
    fn unit_vecs_l_infty() -> UnitVecsLInfty<S, Self> {
        UnitVecsLInfty::new()
    }

    /// Returns the partial ordering by component of two vectors.
    fn componentwise_cmp(&self, other: &Self) -> Option<Ordering>;

    /// Returns the lexicographic total ordering for this and another vector.
    ///
    /// That is, the first different coordinate decides the ordering.
    /// This is useful as an arbitrary total ordering for sorting,
    /// but is not intended to be otherwise meaningful.
    fn lex_cmp(&self, other: &Self) -> Ordering;
}

/// An iterator that yields the unit vectors.
#[derive(Clone, Debug)]
pub struct UnitVecs<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    _s: PhantomData<S>,
    _v: PhantomData<V>,
    i: usize,
}
impl<S, V> UnitVecs<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn new() -> UnitVecs<S, V> {
        UnitVecs {
            _s: PhantomData,
            _v: PhantomData,
            i: 0,
        }
    }
}
impl<S, V> Iterator for UnitVecs<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Item = V;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < V::DIM {
            let v = V::with(|j| if j == self.i { S::one() } else { S::zero() });
            self.i += 1;
            Some(v)
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = V::DIM - self.i;
        (len, Some(len))
    }
}
impl<S, V> ExactSizeIterator for UnitVecs<S, V>
where
    S: Integer,
    V: Vector<S>,
{
}

/// An iterator that yields the vectors to orthogonal neighbours.
///
/// These are the vectors with L1 norm equal to 1.
#[derive(Clone, Debug)]
pub struct UnitVecsL1<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    _s: PhantomData<S>,
    _v: PhantomData<V>,
    i: usize,
}
impl<S, V> UnitVecsL1<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn new() -> UnitVecsL1<S, V> {
        UnitVecsL1 {
            _s: PhantomData,
            _v: PhantomData,
            i: 0,
        }
    }
}
impl<S, V> Iterator for UnitVecsL1<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Item = V;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i / 2;
        if i < V::DIM {
            let signed_one = if self.i % 2 == 0 { S::one() } else { -S::one() };
            let v = V::with(|j| if j == i { signed_one } else { S::zero() });
            self.i += 1;
            Some(v)
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = 2 * V::DIM - self.i;
        (len, Some(len))
    }
}
impl<S, V> ExactSizeIterator for UnitVecsL1<S, V>
where
    S: Integer,
    V: Vector<S>,
{
}

/// An iterator that yields the vectors to orthogonal and diagonal neighbours.
///
/// These are the vectors with L∞ norm equal to 1.
#[derive(Clone, Debug)]
pub struct UnitVecsLInfty<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    _s: PhantomData<S>,
    _v: PhantomData<V>,
    i: usize,
}
impl<S, V> UnitVecsLInfty<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn new() -> UnitVecsLInfty<S, V> {
        UnitVecsLInfty {
            _s: PhantomData,
            _v: PhantomData,
            i: 0,
        }
    }
}
impl<S, V> Iterator for UnitVecsLInfty<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Item = V;
    fn next(&mut self) -> Option<Self::Item> {
        let len = 3_usize.pow(V::DIM as u32) - 1;
        if self.i < len {
            // We skip the zero vector at the half point.
            let half = len / 2;
            let i = if self.i < half { self.i } else { self.i + 1 };

            let v = V::with(|j| {
                let d = i / 3_usize.pow(j as u32) % 3;
                match d {
                    0 => -S::one(),
                    1 => S::zero(),
                    2 => S::one(),
                    _ => panic!("internal error"),
                }
            });
            self.i += 1;
            Some(v)
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = 3_usize.pow(V::DIM as u32) - 1 - self.i;
        (len, Some(len))
    }
}
impl<S, V> ExactSizeIterator for UnitVecsLInfty<S, V>
where
    S: Integer,
    V: Vector<S>,
{
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
