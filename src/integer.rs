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
}

impl IntegerOps for i32 {}
impl IntegerOps for i64 {}

impl Integer for i32 {
    fn abs(self) -> Self {
        i32::abs(self)
    }
}
impl Integer for i64 {
    fn abs(self) -> Self {
        i64::abs(self)
    }
}
