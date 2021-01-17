//! 3-dimensional matrices.

use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

use crate::v3d;
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
    ///
    /// See [`Matrix2d::with`](crate::Matrix2d::with) for an example.
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
    /// Returns a row vector of a matrix.
    pub fn row_vec(&self, i: usize) -> Vec3d<S> {
        v3d(self.a[i][0], self.a[i][1], self.a[i][2])
    }
    /// Returns a column vector of a matrix.
    pub fn col_vec(&self, j: usize) -> Vec3d<S> {
        v3d(self.a[0][j], self.a[1][j], self.a[2][j])
    }
    /// Returns the diagonal vector of a matrix.
    pub fn diag_vec(&self) -> Vec3d<S> {
        v3d(self.a[0][0], self.a[1][1], self.a[2][2])
    }
    /// Returns the determinant of a matrix.
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
        v3d(x, y, z)
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
        v3d(x, y, z)
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
        v3d(x, y, z)
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
        v3d(x, y, z)
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
#[cfg(test)]
mod tests {
    use core::convert::TryFrom;

    use crate::v3d;
    use crate::Matrix3d;

    #[test]
    fn test_zero() {
        let m = Matrix3d::zero();
        assert_eq!(v3d(0, 0, 0), m.row_vec(0));
        assert_eq!(v3d(0, 0, 0), m.row_vec(1));
        assert_eq!(v3d(0, 0, 0), m.row_vec(2));
    }

    #[test]
    fn test_unit() {
        let m = Matrix3d::unit();
        assert_eq!(v3d(1, 0, 0), m.row_vec(0));
        assert_eq!(v3d(0, 1, 0), m.row_vec(1));
        assert_eq!(v3d(0, 0, 1), m.row_vec(2));
    }

    #[test]
    fn test_row_vec() {
        let m = Matrix3d::with(|i, j| i64::try_from(3 * i + j).unwrap());
        assert_eq!(v3d(0, 1, 2), m.row_vec(0));
        assert_eq!(v3d(3, 4, 5), m.row_vec(1));
        assert_eq!(v3d(6, 7, 8), m.row_vec(2));
    }

    #[test]
    fn test_col_vec() {
        let m = Matrix3d::with(|i, j| i64::try_from(3 * i + j).unwrap());
        assert_eq!(v3d(0, 3, 6), m.col_vec(0));
        assert_eq!(v3d(1, 4, 7), m.col_vec(1));
        assert_eq!(v3d(2, 5, 8), m.col_vec(2));
    }

    #[test]
    fn test_diag_vec() {
        let m = Matrix3d::with(|i, j| i64::try_from(3 * i + j).unwrap());
        assert_eq!(v3d(0, 4, 8), m.diag_vec());
    }

    #[test]
    fn test_det() {
        // Test a diagonal matrix and all its row permutations,
        // in order to cover all terms of the determinant formula.

        // The determinant of a diagonal matrix is the product of its diagonal elements
        // (all other elements are zero in a diagonal matrix).
        let m = Matrix3d::with(|i, j| i64::try_from(if i == j { i + 2 } else { 0 }).unwrap());
        assert_eq!(24, m.det());

        // Rotations of the rows don't change the determinant.
        let m =
            Matrix3d::with(|i, j| i64::try_from(if (i + 1) % 3 == j { i + 2 } else { 0 }).unwrap());
        assert_eq!(24, m.det());
        let m =
            Matrix3d::with(|i, j| i64::try_from(if (i + 2) % 3 == j { i + 2 } else { 0 }).unwrap());
        assert_eq!(24, m.det());

        // Transpositions of the rows change the sign.
        // Swap rows 1 and 2, keep row 0 fixed
        let m =
            Matrix3d::with(|i, j| i64::try_from(if (3 - i) % 3 == j { i + 2 } else { 0 }).unwrap());
        assert_eq!(-24, m.det());
        // Swap rows 0 and 2, keep row 1 fixed
        let m =
            Matrix3d::with(|i, j| i64::try_from(if (2 - i) % 3 == j { i + 2 } else { 0 }).unwrap());
        assert_eq!(-24, m.det());
        // Swap rows 0 and 1, keep row 2 fixed
        let m =
            Matrix3d::with(|i, j| i64::try_from(if (4 - i) % 3 == j { i + 2 } else { 0 }).unwrap());
        assert_eq!(-24, m.det());
    }

    #[test]
    fn test_mul_mv() {
        let m = Matrix3d::with(|i, j| i64::try_from(if i == j { i + 2 } else { 0 }).unwrap());
        let v = v3d(2, 3, 4);
        assert_eq!(v3d(4, 9, 16), m * v);
        assert_eq!(v3d(4, 9, 16), m * &v);
        assert_eq!(v3d(4, 9, 16), &m * v);
        assert_eq!(v3d(4, 9, 16), &m * &v);
    }

    #[test]
    fn test_mul_mm() {
        let m = Matrix3d::with(|i, j| i64::try_from(if i == j { i + 2 } else { 0 }).unwrap());
        assert_eq!(v3d(4, 9, 16), (m * m).diag_vec());
        assert_eq!(v3d(4, 9, 16), (m * &m).diag_vec());
        assert_eq!(v3d(4, 9, 16), (&m * m).diag_vec());
        assert_eq!(v3d(4, 9, 16), (&m * &m).diag_vec());
    }
}
