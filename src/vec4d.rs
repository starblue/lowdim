use core::ops::Add;
use core::ops::AddAssign;
use core::ops::Index;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Sub;
use core::ops::SubAssign;

use crate::Integer;
use crate::Vector;
use crate::VectorOps;

const DIM: usize = 4;

/// A four-dimensional discrete vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Vec4d<S = i64>([S; DIM])
where
    S: Integer;

impl<S: Integer> Vec4d<S> {
    /// Creates a new 4d-vector from its coordinates.
    pub fn new(x: S, y: S, z: S, w: S) -> Vec4d<S> {
        Vec4d([x, y, z, w])
    }

    pub fn x(&self) -> S {
        self.0[0]
    }
    pub fn y(&self) -> S {
        self.0[1]
    }
    pub fn z(&self) -> S {
        self.0[2]
    }
    pub fn w(&self) -> S {
        self.0[3]
    }
}

impl<S: Integer> VectorOps<S, Vec4d<S>> for Vec4d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec4d<S>> for Vec4d<S> {}
impl<'a, S: Integer> VectorOps<S, Vec4d<S>, Vec4d<S>> for &'a Vec4d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec4d<S>, Vec4d<S>> for &'a Vec4d<S> {}

impl<S: Integer> Vector<S> for Vec4d<S> {
    fn with<F>(f: F) -> Vec4d<S>
    where
        F: Fn(usize) -> S,
    {
        Vec4d([f(0), f(1), f(2), f(3)])
    }

    /// The L1, taxicab or Manhatten norm.
    fn norm_l1(&self) -> S {
        let abs_x = self.x().abs();
        let abs_y = self.y().abs();
        let abs_z = self.z().abs();
        let abs_w = self.w().abs();
        abs_x + abs_y + abs_z + abs_w
    }
    /// Creates a vector of the 6 orthogonal vectors, i.e. those with L1 norm equal to 1.
    fn unit_vecs_l1() -> Vec<Self> {
        vec![
            v4d(1, 0, 0, 0),
            v4d(0, 1, 0, 0),
            v4d(0, 0, 1, 0),
            v4d(0, 0, 0, 1),
            v4d(-1, 0, 0, 0),
            v4d(0, -1, 0, 0),
            v4d(0, 0, -1, 0),
            v4d(0, 0, 0, -1),
        ]
    }

    /// The maximum, Chebychev or L-infinity norm.
    fn norm_infty(&self) -> S {
        let abs_x = self.x().abs();
        let abs_y = self.y().abs();
        let abs_z = self.z().abs();
        let abs_w = self.w().abs();
        abs_x.max(abs_y).max(abs_z).max(abs_w)
    }
    /// Creates a vector of the 26 vectors with L∞ norm equal to 1.
    fn unit_vecs_l_infty() -> Vec<Self> {
        let mut result = Vec::new();
        for w in -1..=1 {
            for z in -1..=1 {
                for y in -1..=1 {
                    for x in -1..=1 {
                        if x != 0 || y != 0 || z != 0 || w != 0 {
                            result.push(v4d(x, y, z, w));
                        }
                    }
                }
            }
        }
        result
    }
}

/// Creates a 4d-vector.
///
/// This is a utility function for concisely representing vectors.
pub fn v4d<S: Integer, T>(x: T, y: T, z: T, w: T) -> Vec4d<S>
where
    S: From<T>,
{
    Vec4d::new(S::from(x), S::from(y), S::from(z), S::from(w))
}

impl<S: Integer> Index<usize> for Vec4d<S> {
    type Output = S;
    fn index(&self, i: usize) -> &S {
        self.0.index(i)
    }
}

impl<S: Integer> Add<Vec4d<S>> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn add(self, other: Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec4d<S>> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn add(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<Vec4d<S>> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn add(self, other: Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec4d<S>> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn add(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> Sub<Vec4d<S>> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn sub(self, other: Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec4d<S>> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn sub(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<Vec4d<S>> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn sub(self, other: Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec4d<S>> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn sub(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] - other[i])
    }
}

impl<S: Integer> Neg for Vec4d<S> {
    type Output = Vec4d<S>;

    fn neg(self) -> Vec4d<S> {
        Vec4d::with(|i| -self[i])
    }
}
impl<'a, S: Integer> Neg for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn neg(self) -> Vec4d<S> {
        Vec4d::with(|i| -self[i])
    }
}
impl Mul<Vec4d<i64>> for i64 {
    type Output = Vec4d<i64>;

    fn mul(self, other: Vec4d<i64>) -> Vec4d<i64> {
        Vec4d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec4d<i64>> for i64 {
    type Output = Vec4d<i64>;

    fn mul(self, other: &'a Vec4d<i64>) -> Vec4d<i64> {
        Vec4d::with(|i| self * other[i])
    }
}

impl<'a> Mul<Vec4d<i64>> for &'a i64 {
    type Output = Vec4d<i64>;

    fn mul(self, other: Vec4d<i64>) -> Vec4d<i64> {
        Vec4d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec4d<i64>> for &'a i64 {
    type Output = Vec4d<i64>;

    fn mul(self, other: &'a Vec4d<i64>) -> Vec4d<i64> {
        Vec4d::with(|i| self * other[i])
    }
}

impl<S: Integer> Mul<Vec4d<S>> for Vec4d<S> {
    type Output = S;

    fn mul(self, other: Vec4d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z() + self.w() * other.w()
    }
}

impl<'a, S: Integer> Mul<&'a Vec4d<S>> for Vec4d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec4d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z() + self.w() * other.w()
    }
}

impl<'a, S: Integer> Mul<Vec4d<S>> for &'a Vec4d<S> {
    type Output = S;

    fn mul(self, other: Vec4d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z() + self.w() * other.w()
    }
}

impl<'a, S: Integer> Mul<&'a Vec4d<S>> for &'a Vec4d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec4d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z() + self.w() * other.w()
    }
}

impl<S: Integer> AddAssign for Vec4d<S> {
    fn add_assign(&mut self, other: Vec4d<S>) {
        *self = Vec4d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> AddAssign<&'a Vec4d<S>> for Vec4d<S> {
    fn add_assign(&mut self, other: &'a Vec4d<S>) {
        *self = Vec4d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> SubAssign for Vec4d<S> {
    fn sub_assign(&mut self, other: Vec4d<S>) {
        *self = Vec4d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> SubAssign<&'a Vec4d<S>> for Vec4d<S> {
    fn sub_assign(&mut self, other: &'a Vec4d<S>) {
        *self = Vec4d::with(|i| self[i] - other[i])
    }
}

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;

    use crate::v4d;
    use crate::Vec4d;
    use crate::Vector;

    #[test]
    fn test_new_x_y() {
        let v = Vec4d::new(3, 7, 1, -2);
        assert_eq!(3, v.x());
        assert_eq!(7, v.y());
        assert_eq!(1, v.z());
        assert_eq!(-2, v.w());
    }

    #[test]
    fn test_with() {
        assert_eq!(
            v4d(2, 3, 4, 5),
            Vec4d::with(|i| i64::try_from(i + 2).unwrap())
        );
    }

    #[test]
    fn test_norm_l1() {
        assert_eq!(18, v4d(2, 3, 5, 8).norm_l1());
        assert_eq!(18, v4d(-2, -3, -5, -8).norm_l1());
    }
    #[test]
    fn test_norm_infty() {
        assert_eq!(8, v4d(2, 3, 5, 8).norm_infty());
        assert_eq!(8, v4d(-2, -3, -5, -8).norm_infty());
    }

    #[test]
    fn test_index() {
        let v = Vec4d::new(3, 7, 1, -2);
        assert_eq!(3, v[0]);
        assert_eq!(7, v[1]);
        assert_eq!(1, v[2]);
        assert_eq!(-2, v[3]);
    }

    #[test]
    fn test_add() {
        let u = Vec4d::new(2, 1, 5, -4);
        let v = Vec4d::new(3, 7, 1, 4);
        assert_eq!(v4d(5, 8, 6, 0), u + v);
        assert_eq!(v4d(5, 8, 6, 0), u + &v);
        assert_eq!(v4d(5, 8, 6, 0), &u + v);
        assert_eq!(v4d(5, 8, 6, 0), &u + &v);
    }

    #[test]
    fn test_sub() {
        let u = Vec4d::new(2, 1, 5, -4);
        let v = Vec4d::new(3, 7, 1, 4);
        assert_eq!(v4d(-1, -6, 4, -8), u - v);
        assert_eq!(v4d(-1, -6, 4, -8), u - &v);
        assert_eq!(v4d(-1, -6, 4, -8), &u - v);
        assert_eq!(v4d(-1, -6, 4, -8), &u - &v);
    }

    #[test]
    fn test_neg() {
        let u = Vec4d::new(2, 1, -3, 7);
        assert_eq!(v4d(-2, -1, 3, -7), -u);
        assert_eq!(v4d(-2, -1, 3, -7), -&u);
    }

    #[test]
    fn test_mul_sv() {
        let v = Vec4d::new(3, 7, 1, 4);
        assert_eq!(v4d(6, 14, 2, 8), 2 * v);
        assert_eq!(v4d(6, 14, 2, 8), 2 * &v);
        assert_eq!(v4d(6, 14, 2, 8), &2 * v);
        assert_eq!(v4d(6, 14, 2, 8), &2 * &v);
    }

    #[test]
    fn test_mul_vv() {
        let u = Vec4d::new(2, 1, 5, -4);
        let v = Vec4d::new(3, 7, 1, 4);
        assert_eq!(2, u * v);
        assert_eq!(2, u * &v);
        assert_eq!(2, &u * v);
        assert_eq!(2, &u * &v);
    }

    #[test]
    fn test_add_assign() {
        let mut u = Vec4d::new(2, 1, 5, 4);
        u += Vec4d::new(3, 7, 1, -4);
        assert_eq!(v4d(5, 8, 6, 0), u);
        u += &Vec4d::new(3, 7, 1, -4);
        assert_eq!(v4d(8, 15, 7, -4), u);
    }

    #[test]
    fn test_sub_assign() {
        let mut u = Vec4d::new(2, 1, 5, 4);
        u -= Vec4d::new(3, 7, 1, -4);
        assert_eq!(v4d(-1, -6, 4, 8), u);
        u -= &Vec4d::new(3, 7, 1, -4);
        assert_eq!(v4d(-4, -13, 3, 12), u);
    }

    #[test]
    fn test_min() {
        let u = Vec4d::new(2, 1, 5, -4);
        let v = Vec4d::new(3, 7, 1, 4);
        assert_eq!(v4d(2, 1, 1, -4), u.min(v));
    }

    #[test]
    fn test_max() {
        let u = Vec4d::new(2, 1, 5, -4);
        let v = Vec4d::new(3, 7, 1, 4);
        assert_eq!(v4d(3, 7, 5, 4), u.max(v));
    }
}
