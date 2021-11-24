//! Contains traits defining required properties of the underlying integer types.

use core::fmt;
use core::iter;
use core::ops;

/// Required arithmetic operations for integers.
///
/// Must be in a separate trait to allow `Self` to be a reference type
/// and the output the base type.
pub trait IntegerOps
where
    Self: Sized,
    Self: ops::Add<Output = Self> + for<'a> ops::Add<&'a Self, Output = Self>,
    Self: ops::Sub<Output = Self> + for<'a> ops::Sub<&'a Self, Output = Self>,
    Self: ops::Mul<Output = Self> + for<'a> ops::Mul<&'a Self, Output = Self>,
    Self: ops::Div<Output = Self> + for<'a> ops::Div<&'a Self, Output = Self>,
    Self: ops::Neg<Output = Self>,
    Self: ops::AddAssign<Self> + for<'a> ops::AddAssign<&'a Self>,
    Self: ops::SubAssign<Self> + for<'a> ops::SubAssign<&'a Self>,
    Self: ops::MulAssign<Self> + for<'a> ops::MulAssign<&'a Self>,
    Self: ops::DivAssign<Self> + for<'a> ops::DivAssign<&'a Self>,
    Self: ops::RemAssign<Self> + for<'a> ops::RemAssign<&'a Self>,
    Self: iter::Sum<Self> + for<'a> iter::Sum<&'a Self>,
    Self: iter::Product<Self> + for<'a> iter::Product<&'a Self>,
{
}

/// Required traits and operations for integers.
pub trait Integer
where
    Self: Copy,
    Self: fmt::Display,
    Self: TryFrom<usize>,
    Self: Ord,
    Self: IntegerOps,
{
    /// Returns zero.
    fn zero() -> Self;

    /// Returns one.
    fn one() -> Self;

    /// Returns two.
    fn two() -> Self;

    /// The absolute value function.
    fn abs(self) -> Self;

    /// The sign function.
    ///
    /// Returns `1` for positive numbers, `0` for zero and `-1` for negative numbers.
    fn signum(self) -> Self;

    /// Converts to an `usize` if possible, or panics otherwise.
    fn to_usize(self) -> usize
    where
        usize: TryFrom<Self>,
        <usize as TryFrom<Self>>::Error: fmt::Debug,
    {
        usize::try_from(self).unwrap()
    }

    /// Converts from an `usize` if possible, or panics otherwise.
    fn from_usize(n: usize) -> Self
    where
        <Self as TryFrom<usize>>::Error: fmt::Debug,
    {
        Self::try_from(n).unwrap()
    }
}

#[doc(hidden)]
macro_rules! impl_integer {
    ($t:ty) => {
        impl IntegerOps for $t {}

        impl Integer for $t {
            fn zero() -> Self {
                0
            }
            fn one() -> Self {
                1
            }
            fn two() -> Self {
                2
            }
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            fn signum(self) -> Self {
                <$t>::signum(self)
            }
        }
    };
}

impl_integer!(i8);
impl_integer!(i16);
impl_integer!(i32);
impl_integer!(i64);
impl_integer!(i128);
impl_integer!(isize);

#[cfg(test)]
mod tests {
    use crate::Integer;

    #[test]
    fn test_abs_i32() {
        assert_eq!(5, Integer::abs(5_i32));
        assert_eq!(5, Integer::abs(-5_i32));
    }

    #[test]
    fn test_abs_i64() {
        assert_eq!(5, Integer::abs(5_i64));
        assert_eq!(5, Integer::abs(-5_i64));
    }

    #[test]
    fn test_abs_i128() {
        assert_eq!(5, Integer::abs(5_i128));
        assert_eq!(5, Integer::abs(-5_i128));
    }

    #[test]
    fn test_signum_i32() {
        assert_eq!(1, Integer::signum(5_i32));
        assert_eq!(0, Integer::signum(0_i32));
        assert_eq!(-1, Integer::signum(-5_i32));
    }

    #[test]
    fn test_signum_i64() {
        assert_eq!(1, Integer::signum(5_i64));
        assert_eq!(0, Integer::signum(0_i64));
        assert_eq!(-1, Integer::signum(-5_i64));
    }

    #[test]
    fn test_signum_i128() {
        assert_eq!(1, Integer::signum(5_i128));
        assert_eq!(0, Integer::signum(0_i128));
        assert_eq!(-1, Integer::signum(-5_i128));
    }
}
