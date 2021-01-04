use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

use crate::Integer;
use crate::Vec3d;

const DIM: usize = 3;

/// A two-dimensional discrete matrix.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Matrix3d<S: Integer> {
    a: [[S; DIM]; DIM],
}

impl<S: Integer> Matrix3d<S> {
    /// Creates a new 3d matrix from a function.
    pub fn with<F>(f: F) -> Matrix3d<S>
    where
        F: Fn(usize, usize) -> S,
    {
        let r0 = [f(0, 0), f(0, 1), f(0, 2)];
        let r1 = [f(1, 0), f(1, 1), f(1, 2)];
        let r2 = [f(2, 0), f(2, 1), f(2, 2)];
        Matrix3d { a: [r0, r1, r2] }
    }
    /// Creates a zero matrix.
    pub fn zero() -> Matrix3d<S>
    where
        S: From<i32>,
    {
        Matrix3d::with(|_i, _j| 0.into())
    }
    /// Creates a unit matrix.
    pub fn unit() -> Matrix3d<S>
    where
        S: From<i32>,
    {
        Matrix3d::with(|i, j| if i == j { 1.into() } else { 0.into() })
    }
    /// Accesses a row vector
    pub fn row_vec(&self, i: usize) -> Vec3d<S> {
        Vec3d::new(self.a[i][0], self.a[i][1], self.a[i][2])
    }
    /// Accesses a column vector
    pub fn col_vec(&self, j: usize) -> Vec3d<S> {
        Vec3d::new(self.a[0][j], self.a[1][j], self.a[2][j])
    }
    /// The determinant of the matrix
    pub fn det(&self) -> S
    where
        S: Copy + Sub<Output = S> + Mul<Output = S>,
    {
        self.a[0][0] * self.a[1][1] * self.a[2][2]
            + self.a[0][1] * self.a[1][2] * self.a[2][0]
            + self.a[0][2] * self.a[1][0] * self.a[2][1]
            - self.a[0][2] * self.a[1][1] * self.a[2][0]
            - self.a[0][1] * self.a[1][0] * self.a[2][2]
            - self.a[0][0] * self.a[1][2] * self.a[2][1]
    }
}

impl<S: Integer> Mul<Vec3d<S>> for Matrix3d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec3d<S>;

    fn mul(self, other: Vec3d<S>) -> Vec3d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        let z = self.row_vec(2) * other;
        Vec3d::new(x, y, z)
    }
}

impl<'a, S: Integer> Mul<&'a Vec3d<S>> for Matrix3d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec3d<S>;

    fn mul(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        let z = self.row_vec(2) * other;
        Vec3d::new(x, y, z)
    }
}

impl<'a, S: Integer> Mul<Vec3d<S>> for &'a Matrix3d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec3d<S>;

    fn mul(self, other: Vec3d<S>) -> Vec3d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        let z = self.row_vec(2) * other;
        Vec3d::new(x, y, z)
    }
}

impl<'a, S: Integer> Mul<&'a Vec3d<S>> for &'a Matrix3d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Vec3d<S>;

    fn mul(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        let x = self.row_vec(0) * other;
        let y = self.row_vec(1) * other;
        let z = self.row_vec(2) * other;
        Vec3d::new(x, y, z)
    }
}

impl<S: Integer> Mul<Matrix3d<S>> for Matrix3d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix3d<S>;

    fn mul(self, other: Matrix3d<S>) -> Matrix3d<S> {
        Matrix3d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}

impl<'a, S: Integer> Mul<&'a Matrix3d<S>> for Matrix3d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix3d<S>;

    fn mul(self, other: &'a Matrix3d<S>) -> Matrix3d<S> {
        Matrix3d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}

impl<'a, S: Integer> Mul<Matrix3d<S>> for &'a Matrix3d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix3d<S>;

    fn mul(self, other: Matrix3d<S>) -> Matrix3d<S> {
        Matrix3d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}

impl<'a, S: Integer> Mul<&'a Matrix3d<S>> for &'a Matrix3d<S>
where
    S: Copy + Add<Output = S> + Mul<Output = S>,
{
    type Output = Matrix3d<S>;

    fn mul(self, other: &'a Matrix3d<S>) -> Matrix3d<S> {
        Matrix3d::with(|i, j| self.row_vec(i) * other.col_vec(j))
    }
}
