//! 2-dimensional matrices.

#![warn(missing_docs)]

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
    pub fn new(a: S, b: S, c: S, d: S) -> Matrix2d<S> {
        let r0 = [a, b];
        let r1 = [c, d];
        Matrix2d { a: [r0, r1] }
    }
    /// Creates a new 2d matrix from a function.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::convert::TryFrom;
    /// # use gamedim::Matrix2d;
    /// let m = Matrix2d::with(|i, j| i64::try_from(3 * i + j).unwrap());
    /// assert_eq!(Matrix2d::new(0, 1, 3, 4), m);
    /// ```
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
    /// Returns a row vector of a matrix.
    pub fn row_vec(&self, i: usize) -> Vec2d<S> {
        v2d(self.a[i][0], self.a[i][1])
    }
    /// Returns a column vector of a matrix.
    pub fn col_vec(&self, j: usize) -> Vec2d<S> {
        v2d(self.a[0][j], self.a[1][j])
    }
    /// Returns the diagonal vector of a matrix.
    pub fn diag_vec(&self) -> Vec2d<S> {
        v2d(self.a[0][0], self.a[1][1])
    }
    /// Returns the determinant of a matrix.
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
        Matrix2d::new(S::from(0), S::from(-1), S::from(1), S::from(0))
    }
    /// Creates a matrix for a right rotation by a right angle.
    pub fn rotate_right_90() -> Matrix2d<S>
    where
        S: From<i32>,
    {
        Matrix2d::new(S::from(0), S::from(1), S::from(-1), S::from(0))
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

#[cfg(test)]
mod tests {
    use core::convert::TryFrom;

    use crate::v2d;
    use crate::Matrix2d;

    #[test]
    fn test_with() {
        let m = Matrix2d::with(|i, j| i64::try_from(3 * i + j).unwrap());
        assert_eq!(Matrix2d::new(0, 1, 3, 4), m);
    }

    #[test]
    fn test_zero() {
        let m = Matrix2d::zero();
        assert_eq!(Matrix2d::new(0, 0, 0, 0), m);
    }

    #[test]
    fn test_unit() {
        let m = Matrix2d::unit();
        assert_eq!(Matrix2d::new(1, 0, 0, 1), m);
    }

    #[test]
    fn test_row_vec() {
        let m = Matrix2d::new(1, 2, 3, 4);
        assert_eq!(v2d(1, 2), m.row_vec(0));
    }

    #[test]
    fn test_col_vec() {
        let m = Matrix2d::new(1, 2, 3, 4);
        assert_eq!(v2d(2, 4), m.col_vec(1));
    }

    #[test]
    fn test_diag_vec() {
        let m = Matrix2d::with(|i, j| i64::try_from(3 * i + j).unwrap());
        assert_eq!(v2d(0, 4), m.diag_vec());
    }

    #[test]
    fn test_det() {
        // Test a diagonal matrix and all its row permutations,
        // in order to cover all terms of the determinant formula.

        // The determinant of a diagonal matrix is the product of its diagonal elements
        // (all other elements are zero in a diagonal matrix).
        let m = Matrix2d::new(2, 0, 0, 3);
        assert_eq!(6, m.det());

        // Transpositions of the rows change the sign.
        let m = Matrix2d::new(0, 3, 2, 0);
        assert_eq!(-6, m.det());
    }

    #[test]
    fn test_rotate_left_90() {
        let vx = v2d(1, 0);
        let vy = v2d(0, 1);
        let m = Matrix2d::rotate_left_90();
        assert_eq!(vy, m * vx);
        assert_eq!(-vx, m * vy);
    }

    #[test]
    fn test_rotate_right_90() {
        let vx = v2d(1, 0);
        let vy = v2d(0, 1);
        let m = Matrix2d::rotate_right_90();
        assert_eq!(-vy, m * vx);
        assert_eq!(vx, m * vy);
    }

    #[test]
    fn test_mul_mv() {
        let m = Matrix2d::new(1, 3, 7, 15);
        let v = v2d(2, 3);
        assert_eq!(v2d(11, 59), m * v);
        assert_eq!(v2d(11, 59), m * &v);
        assert_eq!(v2d(11, 59), &m * v);
        assert_eq!(v2d(11, 59), &m * &v);
    }

    #[test]
    fn test_mul_mm() {
        let m0 = Matrix2d::new(1, 3, 7, 15);
        let m1 = Matrix2d::new(1, 2, 3, 4);
        assert_eq!(Matrix2d::new(10, 14, 52, 74), m0 * m1);
        assert_eq!(Matrix2d::new(10, 14, 52, 74), m0 * &m1);
        assert_eq!(Matrix2d::new(10, 14, 52, 74), &m0 * m1);
        assert_eq!(Matrix2d::new(10, 14, 52, 74), &m0 * &m1);
    }
}
