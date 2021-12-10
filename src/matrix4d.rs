//! 4d matrices.

use std::ops::Add;
use std::ops::Mul;

use crate::v4d;
use crate::Integer;
use crate::Matrix;
use crate::MatrixOps;
use crate::Vec4d;

const DIM: usize = 4;

/// A 4d discrete matrix.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Matrix4d<S: Integer> {
    a: [[S; DIM]; DIM],
}

impl<S: Integer> Matrix4d<S> {
    /// Returns a row vector of a matrix.
    pub fn row_vec(&self, i: usize) -> Vec4d<S> {
        v4d(self.a[i][0], self.a[i][1], self.a[i][2], self.a[i][3])
    }
    /// Returns a column vector of a matrix.
    pub fn col_vec(&self, j: usize) -> Vec4d<S> {
        v4d(self.a[0][j], self.a[1][j], self.a[2][j], self.a[3][j])
    }
    /// Returns the diagonal vector of a matrix.
    pub fn diag_vec(&self) -> Vec4d<S> {
        v4d(self.a[0][0], self.a[1][1], self.a[2][2], self.a[3][3])
    }
}

impl<S: Integer> MatrixOps<S, Matrix4d<S>> for Matrix4d<S> {}
impl<'a, S: Integer> MatrixOps<S, &'a Matrix4d<S>> for Matrix4d<S> {}
impl<'a, S: Integer> MatrixOps<S, Matrix4d<S>, Matrix4d<S>> for &'a Matrix4d<S> {}
impl<'a, S: Integer> MatrixOps<S, &'a Matrix4d<S>, Matrix4d<S>> for &'a Matrix4d<S> {}

impl<S: Integer> MatrixOps<S, Vec4d<S>, Vec4d<S>> for Matrix4d<S> {}
impl<'a, S: Integer> MatrixOps<S, &'a Vec4d<S>, Vec4d<S>> for Matrix4d<S> {}
impl<'a, S: Integer> MatrixOps<S, Vec4d<S>, Vec4d<S>> for &'a Matrix4d<S> {}
impl<'a, S: Integer> MatrixOps<S, &'a Vec4d<S>, Vec4d<S>> for &'a Matrix4d<S> {}

impl<S: Integer> Matrix<S> for Matrix4d<S> {
    type V = Vec4d<S>;

    fn with<F>(f: F) -> Matrix4d<S>
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
        v4d(x, y, z, w)
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
        v4d(x, y, z, w)
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
        v4d(x, y, z, w)
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
        v4d(x, y, z, w)
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
#[cfg(test)]
mod tests {
    use core::convert::TryFrom;

    use crate::v4d;
    use crate::Matrix;
    use crate::Matrix4d;

    #[test]
    fn test_zero() {
        let m = Matrix4d::zero();
        assert_eq!(v4d(0, 0, 0, 0), m.row_vec(0));
        assert_eq!(v4d(0, 0, 0, 0), m.row_vec(1));
        assert_eq!(v4d(0, 0, 0, 0), m.row_vec(2));
        assert_eq!(v4d(0, 0, 0, 0), m.row_vec(3));
    }

    #[test]
    fn test_unit() {
        let m = Matrix4d::unit();
        assert_eq!(v4d(1, 0, 0, 0), m.row_vec(0));
        assert_eq!(v4d(0, 1, 0, 0), m.row_vec(1));
        assert_eq!(v4d(0, 0, 1, 0), m.row_vec(2));
        assert_eq!(v4d(0, 0, 0, 1), m.row_vec(3));
    }

    #[test]
    fn test_row_vec() {
        let m = Matrix4d::with(|i, j| i64::try_from(4 * i + j).unwrap());
        assert_eq!(v4d(0, 1, 2, 3), m.row_vec(0));
        assert_eq!(v4d(4, 5, 6, 7), m.row_vec(1));
        assert_eq!(v4d(8, 9, 10, 11), m.row_vec(2));
        assert_eq!(v4d(12, 13, 14, 15), m.row_vec(3));
    }

    #[test]
    fn test_col_vec() {
        let m = Matrix4d::with(|i, j| i64::try_from(4 * i + j).unwrap());
        assert_eq!(v4d(0, 4, 8, 12), m.col_vec(0));
        assert_eq!(v4d(1, 5, 9, 13), m.col_vec(1));
        assert_eq!(v4d(2, 6, 10, 14), m.col_vec(2));
        assert_eq!(v4d(3, 7, 11, 15), m.col_vec(3));
    }

    #[test]
    fn test_diag_vec() {
        let m = Matrix4d::with(|i, j| i64::try_from(4 * i + j).unwrap());
        assert_eq!(v4d(0, 5, 10, 15), m.diag_vec());
    }

    #[test]
    fn test_mul_mv() {
        let m = Matrix4d::with(|i, j| i64::try_from(if i == j { i + 2 } else { 0 }).unwrap());
        let v = v4d(2, 3, 4, 5);
        assert_eq!(v4d(4, 9, 16, 25), m * v);
        assert_eq!(v4d(4, 9, 16, 25), m * &v);
        assert_eq!(v4d(4, 9, 16, 25), &m * v);
        assert_eq!(v4d(4, 9, 16, 25), &m * &v);
    }

    #[test]
    fn test_mul_mm() {
        let m = Matrix4d::with(|i, j| i64::try_from(if i == j { i + 2 } else { 0 }).unwrap());
        assert_eq!(v4d(4, 9, 16, 25), (m * m).diag_vec());
        assert_eq!(v4d(4, 9, 16, 25), (m * &m).diag_vec());
        assert_eq!(v4d(4, 9, 16, 25), (&m * m).diag_vec());
        assert_eq!(v4d(4, 9, 16, 25), (&m * &m).diag_vec());
    }
}
