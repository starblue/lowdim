use core::ops;

pub trait IntegerOps
where
    Self: Sized,
    Self: ops::Add<Output = Self> + for<'a> ops::Add<&'a Self, Output = Self>,
    Self: ops::Sub<Output = Self> + for<'a> ops::Sub<&'a Self, Output = Self>,
    Self: ops::Mul<Output = Self> + for<'a> ops::Mul<&'a Self, Output = Self>,
    Self: ops::Neg<Output = Self>,
{
}

pub trait Integer
where
    Self: Copy,
    Self: From<i32>,
    Self: Ord,
    Self: IntegerOps,
{
    fn abs(self) -> Self;
    fn signum(self) -> Self;
}

impl IntegerOps for i32 {}
impl IntegerOps for i64 {}

impl Integer for i32 {
    fn abs(self) -> Self {
        i32::abs(self)
    }
    fn signum(self) -> Self {
        i32::signum(self)
    }
}
impl Integer for i64 {
    fn abs(self) -> Self {
        i64::abs(self)
    }
    fn signum(self) -> Self {
        i64::signum(self)
    }
}

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
}
