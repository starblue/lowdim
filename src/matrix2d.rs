use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

use crate::v2d;
use crate::Integer;
use crate::Vec2d;

const DIM: usize = 2;

/// A two-dimensional discrete matrix.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Matrix2d<S: Integer> {
    a: [[S; DIM]; DIM],
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
        let r0 = [S::from(a), S::from(b)];
        let r1 = [S::from(c), S::from(d)];
        Matrix2d { a: [r0, r1] }
    }
    /// Creates a new 2d matrix from a function.
    pub fn with<F>(f: F) -> Matrix2d<S>
    where
        F: Fn(usize, usize) -> S,
    {
        let r0 = [f(0, 0), f(0, 1)];
        let r1 = [f(1, 0), f(1, 1)];
        Matrix2d { a: [r0, r1] }
    }
    /// Creates a zero matrix.
    pub fn zero() -> Matrix2d<S>
    where
        S: From<i32>,
    {
        Matrix2d::with(|_i, _j| 0.into())
    }
    /// Creates a unit matrix.
    pub fn unit() -> Matrix2d<S>
    where
        S: From<i32>,
    {
        Matrix2d::with(|i, j| if i == j { 1.into() } else { 0.into() })
    }
    /// Accesses a row vector
    pub fn row_vec(&self, i: usize) -> Vec2d<S> {
        v2d(self.a[i][0], self.a[i][1])
    }
    /// Accesses a column vector
    pub fn col_vec(&self, j: usize) -> Vec2d<S> {
        v2d(self.a[0][j], self.a[1][j])
    }
    /// The determinant of the matrix
    pub fn det(&self) -> S
    where
        S: Copy + Sub<Output = S> + Mul<Output = S>,
    {
        self.a[0][0] * self.a[1][1] - self.a[0][1] * self.a[1][0]
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
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        v2d(x, y)
    }
}

impl<'a, S: Integer> Mul<&'a Vec2d<S>> for Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec2d<S>;

    fn mul(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        v2d(x, y)
    }
}

impl<'a, S: Integer> Mul<Vec2d<S>> for &'a Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec2d<S>;

    fn mul(self, other: Vec2d<S>) -> Vec2d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        v2d(x, y)
    }
}

impl<'a, S: Integer> Mul<&'a Vec2d<S>> for &'a Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec2d<S>;

    fn mul(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        v2d(x, y)
    }
}

impl<S: Integer> Mul<Matrix2d<S>> for Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix2d<S>;

    fn mul(self, other: Matrix2d<S>) -> Matrix2d<S> {
        Matrix2d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}

impl<'a, S: Integer> Mul<&'a Matrix2d<S>> for Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix2d<S>;

    fn mul(self, other: &'a Matrix2d<S>) -> Matrix2d<S> {
        Matrix2d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}

impl<'a, S: Integer> Mul<Matrix2d<S>> for &'a Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix2d<S>;

    fn mul(self, other: Matrix2d<S>) -> Matrix2d<S> {
        Matrix2d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}

impl<'a, S: Integer> Mul<&'a Matrix2d<S>> for &'a Matrix2d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix2d<S>;

    fn mul(self, other: &'a Matrix2d<S>) -> Matrix2d<S> {
        Matrix2d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}
