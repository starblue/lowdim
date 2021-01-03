use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

use crate::Integer;
use crate::Vec2d;

/// A two-dimensional discrete matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Matrix2d<S: Integer> {
    pub a: S,
    pub b: S,
    pub c: S,
    pub d: S,
}

impl<S: Integer> Matrix2d<S> {
    /// Creates a new 2d matrix from its values by rows.
    ///
    /// (a b)
    /// (c d)
    pub fn new<T>(a: T, b: T, c: T, d: T) -> Matrix2d<S>
    where
        S: From<T>,
    {
        let a = S::from(a);
        let b = S::from(b);
        let c = S::from(c);
        let d = S::from(d);
        Matrix2d { a, b, c, d }
    }
    /// Creates a zero matrix.
    pub fn zero() -> Matrix2d<S>
    where
        S: From<i32>,
    {
        Matrix2d::new(0, 0, 0, 0)
    }
    /// Creates a unit matrix.
    pub fn unit() -> Matrix2d<S>
    where
        S: From<i32>,
    {
        Matrix2d::new(1, 0, 0, 1)
    }
    /// The determinant of the matrix
    pub fn det(&self) -> S
    where
        S: Copy + Sub<Output = S> + Mul<Output = S>,
    {
        self.a * self.d - self.b * self.c
    }
    /// Creates a matrix for a left rotation by a right angle.
    pub fn rotate_left_90() -> Matrix2d<S>
    where
        S: From<i32>,
    {
        Matrix2d::new(0, -1, 1, 0)
    }
    /// Creates a matrix for a right rotation by a right angle.
    pub fn rotate_right_90() -> Matrix2d<S>
    where
        S: From<i32>,
    {
        Matrix2d::new(0, 1, -1, 0)
    }
}

impl<S: Integer> Mul<Vec2d<S>> for Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec2d<S>;

    fn mul(self, other: Vec2d<S>) -> Vec2d<S> {
        let x = self.a * other.x() + self.b * other.y();
        let y = self.c * other.x() + self.d * other.y();
        Vec2d::new(x, y)
    }
}

impl<'a, S: Integer> Mul<&'a Vec2d<S>> for Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec2d<S>;

    fn mul(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        let x = self.a * other.x() + self.b * other.y();
        let y = self.c * other.x() + self.d * other.y();
        Vec2d::new(x, y)
    }
}

impl<'a, S: Integer> Mul<Vec2d<S>> for &'a Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec2d<S>;

    fn mul(self, other: Vec2d<S>) -> Vec2d<S> {
        let x = self.a * other.x() + self.b * other.y();
        let y = self.c * other.x() + self.d * other.y();
        Vec2d::new(x, y)
    }
}

impl<'a, S: Integer> Mul<&'a Vec2d<S>> for &'a Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec2d<S>;

    fn mul(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        let x = self.a * other.x() + self.b * other.y();
        let y = self.c * other.x() + self.d * other.y();
        Vec2d::new(x, y)
    }
}

impl<S: Integer> Mul<Matrix2d<S>> for Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix2d<S>;

    fn mul(self, other: Matrix2d<S>) -> Matrix2d<S> {
        let a = self.a * other.a + self.b * other.c;
        let b = self.a * other.b + self.b * other.d;
        let c = self.c * other.a + self.d * other.c;
        let d = self.c * other.b + self.d * other.d;
        Matrix2d { a, b, c, d }
    }
}

impl<'a, S: Integer> Mul<&'a Matrix2d<S>> for Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix2d<S>;

    fn mul(self, other: &'a Matrix2d<S>) -> Matrix2d<S> {
        let a = self.a * other.a + self.b * other.c;
        let b = self.a * other.b + self.b * other.d;
        let c = self.c * other.a + self.d * other.c;
        let d = self.c * other.b + self.d * other.d;
        Matrix2d { a, b, c, d }
    }
}

impl<'a, S: Integer> Mul<Matrix2d<S>> for &'a Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix2d<S>;

    fn mul(self, other: Matrix2d<S>) -> Matrix2d<S> {
        let a = self.a * other.a + self.b * other.c;
        let b = self.a * other.b + self.b * other.d;
        let c = self.c * other.a + self.d * other.c;
        let d = self.c * other.b + self.d * other.d;
        Matrix2d { a, b, c, d }
    }
}

impl<'a, S: Integer> Mul<&'a Matrix2d<S>> for &'a Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix2d<S>;

    fn mul(self, other: &'a Matrix2d<S>) -> Matrix2d<S> {
        let a = self.a * other.a + self.b * other.c;
        let b = self.a * other.b + self.b * other.d;
        let c = self.c * other.a + self.d * other.c;
        let d = self.c * other.b + self.d * other.d;
        Matrix2d { a, b, c, d }
    }
}
