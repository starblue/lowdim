use std::ops::Add;
use std::ops::Mul;

use crate::Integer;
use crate::Vec4d;

const DIM: usize = 4;

/// A two-dimensional discrete matrix.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Matrix4d<S: Integer> {
    a: [[S; DIM]; DIM],
}

impl<S: Integer> Matrix4d<S> {
    /// Creates a new 4d matrix from a function.
    pub fn with<F>(f: F) -> Matrix4d<S>
    where
        F: Fn(usize, usize) -> S,
    {
        let r0 = [f(0, 0), f(0, 1), f(0, 2), f(0, 3)];
        let r1 = [f(1, 0), f(1, 1), f(1, 2), f(1, 3)];
        let r2 = [f(2, 0), f(2, 1), f(2, 2), f(2, 3)];
        let r3 = [f(3, 0), f(3, 1), f(3, 2), f(3, 3)];
        Matrix4d {
            a: [r0, r1, r2, r3],
        }
    }
    /// Creates a zero matrix.
    pub fn zero() -> Matrix4d<S>
    where
        S: From<i32>,
    {
        Matrix4d::with(|_i, _j| 0.into())
    }
    /// Creates a unit matrix.
    pub fn unit() -> Matrix4d<S>
    where
        S: From<i32>,
    {
        Matrix4d::with(|i, j| if i == j { 1.into() } else { 0.into() })
    }
    /// Accesses a row vector
    pub fn row_vec(&self, i: usize) -> Vec4d<S> {
        Vec4d::new(self.a[i][0], self.a[i][1], self.a[i][2], self.a[i][3])
    }
    /// Accesses a column vector
    pub fn col_vec(&self, j: usize) -> Vec4d<S> {
        Vec4d::new(self.a[0][j], self.a[1][j], self.a[2][j], self.a[3][j])
    }
}

impl<S: Integer> Mul<Vec4d<S>> for Matrix4d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec4d<S>;

    fn mul(self, other: Vec4d<S>) -> Vec4d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        let z = self.row_vec(2) * other;
        let w = self.row_vec(3) * other;
        Vec4d::new(x, y, z, w)
    }
}

impl<'a, S: Integer> Mul<&'a Vec4d<S>> for Matrix4d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec4d<S>;

    fn mul(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        let z = self.row_vec(2) * other;
        let w = self.row_vec(3) * other;
        Vec4d::new(x, y, z, w)
    }
}

impl<'a, S: Integer> Mul<Vec4d<S>> for &'a Matrix4d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec4d<S>;

    fn mul(self, other: Vec4d<S>) -> Vec4d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        let z = self.row_vec(2) * other;
        let w = self.row_vec(3) * other;
        Vec4d::new(x, y, z, w)
    }
}

impl<'a, S: Integer> Mul<&'a Vec4d<S>> for &'a Matrix4d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec4d<S>;

    fn mul(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        let z = self.row_vec(2) * other;
        let w = self.row_vec(3) * other;
        Vec4d::new(x, y, z, w)
    }
}

impl<S: Integer> Mul<Matrix4d<S>> for Matrix4d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix4d<S>;

    fn mul(self, other: Matrix4d<S>) -> Matrix4d<S> {
        Matrix4d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}

impl<'a, S: Integer> Mul<&'a Matrix4d<S>> for Matrix4d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix4d<S>;

    fn mul(self, other: &'a Matrix4d<S>) -> Matrix4d<S> {
        Matrix4d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}

impl<'a, S: Integer> Mul<Matrix4d<S>> for &'a Matrix4d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix4d<S>;

    fn mul(self, other: Matrix4d<S>) -> Matrix4d<S> {
        Matrix4d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}

impl<'a, S: Integer> Mul<&'a Matrix4d<S>> for &'a Matrix4d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix4d<S>;

    fn mul(self, other: &'a Matrix4d<S>) -> Matrix4d<S> {
        Matrix4d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}
