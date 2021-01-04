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
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl IntegerOps for i32 {}
impl IntegerOps for i64 {}

impl Integer for i32 {
    fn abs(self) -> Self {
        i32::abs(self)
    }
    fn min(self, other: i32) -> Self {
        Ord::min(self, other)
    }
    fn max(self, other: i32) -> Self {
        Ord::max(self, other)
    }
}
impl Integer for i64 {
    fn abs(self) -> Self {
        i64::abs(self)
    }
    fn min(self, other: i64) -> Self {
        Ord::min(self, other)
    }
    fn max(self, other: i64) -> Self {
        Ord::max(self, other)
    }
}
