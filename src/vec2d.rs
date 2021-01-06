use core::cmp::Ordering;
use core::ops::Add;
use core::ops::AddAssign;
use core::ops::Index;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Sub;
use core::ops::SubAssign;

use crate::partial_then;
use crate::Integer;
use crate::Vector;
use crate::VectorOps;

const DIM: usize = 2;

/// A two-dimensional discrete vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Vec2d<S = i64>([S; DIM])
where
    S: Integer;

impl<S: Integer> Vec2d<S> {
    /// Creates a new 2d-vector from its coordinates.
    pub fn new(x: S, y: S) -> Vec2d<S> {
        Vec2d([x, y])
    }

    pub fn x(&self) -> S {
        self.0[0]
    }
    pub fn y(&self) -> S {
        self.0[1]
    }

    pub fn rotate_left(&self) -> Self {
        Vec2d::new(-self.y(), self.x())
    }
    pub fn rotate_right(&self) -> Self {
        Vec2d::new(self.y(), -self.x())
    }
}

impl<S: Integer> VectorOps<S, Vec2d<S>> for Vec2d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec2d<S>> for Vec2d<S> {}
impl<'a, S: Integer> VectorOps<S, Vec2d<S>, Vec2d<S>> for &'a Vec2d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec2d<S>, Vec2d<S>> for &'a Vec2d<S> {}

impl<S: Integer> Vector<S> for Vec2d<S> {
    fn with<F>(f: F) -> Vec2d<S>
    where
        F: Fn(usize) -> S,
    {
        Vec2d([f(0), f(1)])
    }

    /// The L1, taxicab or Manhatten norm.
    fn norm_l1(&self) -> S {
        let abs_x = self.x().abs();
        let abs_y = self.y().abs();
        abs_x + abs_y
    }
    /// Creates a vector of the 4 orthogonal vectors, i.e. those with L1 norm equal to 1.
    fn unit_vecs_l1() -> Vec<Self> {
        vec![v2d(1, 0), v2d(0, 1), v2d(-1, 0), v2d(0, -1)]
    }

    /// The maximum, Chebychev or L∞ norm.
    fn norm_infty(&self) -> S {
        let abs_x = self.x().abs();
        let abs_y = self.y().abs();
        abs_x.max(abs_y)
    }
    /// Creates a vector of the 8 vectors with L∞ norm equal to 1.
    fn unit_vecs_l_infty() -> Vec<Self> {
        let mut result = Vec::new();
        for y in -1..=1 {
            for x in -1..=1 {
                if x != 0 || y != 0 {
                    result.push(v2d(x, y));
                }
            }
        }
        result
    }
}

/// Creates a 2d-vector.
///
/// This is a utility function for concisely representing vectors.
pub fn v2d<S: Integer, T>(x: T, y: T) -> Vec2d<S>
where
    S: From<T>,
{
    Vec2d::new(S::from(x), S::from(y))
}

impl<S: Integer> Index<usize> for Vec2d<S> {
    type Output = S;
    fn index(&self, i: usize) -> &S {
        self.0.index(i)
    }
}

impl<S: Integer> PartialOrd for Vec2d<S> {
    fn partial_cmp(&self, other: &Vec2d<S>) -> Option<Ordering> {
        let x_ordering = Some(self.x().cmp(&other.x()));
        let y_ordering = Some(self.y().cmp(&other.y()));
        partial_then(x_ordering, y_ordering)
    }
}

impl<S: Integer> Add<Vec2d<S>> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn add(self, other: Vec2d<S>) -> Vec2d<S> {
        Vec2d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec2d<S>> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn add(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        Vec2d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<Vec2d<S>> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn add(self, other: Vec2d<S>) -> Vec2d<S> {
        Vec2d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec2d<S>> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn add(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        Vec2d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> Sub<Vec2d<S>> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn sub(self, other: Vec2d<S>) -> Vec2d<S> {
        Vec2d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec2d<S>> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn sub(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        Vec2d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<Vec2d<S>> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn sub(self, other: Vec2d<S>) -> Vec2d<S> {
        Vec2d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec2d<S>> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn sub(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        Vec2d::with(|i| self[i] - other[i])
    }
}

impl<S: Integer> Neg for Vec2d<S> {
    type Output = Vec2d<S>;

    fn neg(self) -> Vec2d<S> {
        Vec2d::with(|i| -self[i])
    }
}
impl<'a, S: Integer> Neg for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn neg(self) -> Vec2d<S> {
        Vec2d::with(|i| -self[i])
    }
}

impl Mul<Vec2d<i64>> for i64 {
    type Output = Vec2d<i64>;

    fn mul(self, other: Vec2d<i64>) -> Vec2d<i64> {
        Vec2d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec2d<i64>> for i64 {
    type Output = Vec2d<i64>;

    fn mul(self, other: &'a Vec2d<i64>) -> Vec2d<i64> {
        Vec2d::with(|i| self * other[i])
    }
}

impl<'a> Mul<Vec2d<i64>> for &'a i64 {
    type Output = Vec2d<i64>;

    fn mul(self, other: Vec2d<i64>) -> Vec2d<i64> {
        Vec2d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec2d<i64>> for &'a i64 {
    type Output = Vec2d<i64>;

    fn mul(self, other: &'a Vec2d<i64>) -> Vec2d<i64> {
        Vec2d::with(|i| self * other[i])
    }
}

impl<S: Integer> Mul<Vec2d<S>> for Vec2d<S> {
    type Output = S;

    fn mul(self, other: Vec2d<S>) -> S {
        self.x() * other.x() + self.y() * other.y()
    }
}

impl<'a, S: Integer> Mul<&'a Vec2d<S>> for Vec2d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec2d<S>) -> S {
        self.x() * other.x() + self.y() * other.y()
    }
}

impl<'a, S: Integer> Mul<Vec2d<S>> for &'a Vec2d<S> {
    type Output = S;

    fn mul(self, other: Vec2d<S>) -> S {
        self.x() * other.x() + self.y() * other.y()
    }
}

impl<'a, S: Integer> Mul<&'a Vec2d<S>> for &'a Vec2d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec2d<S>) -> S {
        self.x() * other.x() + self.y() * other.y()
    }
}

impl<S: Integer> AddAssign for Vec2d<S> {
    fn add_assign(&mut self, other: Vec2d<S>) {
        *self = Vec2d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> AddAssign<&'a Vec2d<S>> for Vec2d<S> {
    fn add_assign(&mut self, other: &'a Vec2d<S>) {
        *self = Vec2d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> SubAssign for Vec2d<S> {
    fn sub_assign(&mut self, other: Vec2d<S>) {
        *self = Vec2d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> SubAssign<&'a Vec2d<S>> for Vec2d<S> {
    fn sub_assign(&mut self, other: &'a Vec2d<S>) {
        *self = Vec2d::with(|i| self[i] - other[i])
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;
    use core::convert::TryFrom;

    use crate::v2d;
    use crate::Vec2d;
    use crate::Vector;

    #[test]
    fn test_new_x_y() {
        let v = Vec2d::new(3, 7);
        assert_eq!(3, v.x());
        assert_eq!(7, v.y());
    }

    #[test]
    fn test_rotate_left() {
        let v: Vec2d<i64> = v2d(1, 2);
        assert_eq!(v2d(-2, 1), v.rotate_left());
    }
    #[test]
    fn test_rotate_right() {
        let v: Vec2d<i64> = v2d(1, 2);
        assert_eq!(v2d(2, -1), v.rotate_right());
    }

    #[test]
    fn test_with() {
        assert_eq!(v2d(2, 3), Vec2d::with(|i| i64::try_from(i + 2).unwrap()));
    }

    #[test]
    fn test_norm_l1() {
        assert_eq!(5, v2d(2, 3).norm_l1());
        assert_eq!(5, v2d(-2, 3).norm_l1());
        assert_eq!(5, v2d(2, -3).norm_l1());
        assert_eq!(5, v2d(-2, -3).norm_l1());
    }
    #[test]
    fn test_norm_infty() {
        assert_eq!(3, v2d(2, 3).norm_infty());
        assert_eq!(3, v2d(-2, 3).norm_infty());
        assert_eq!(3, v2d(2, -3).norm_infty());
        assert_eq!(3, v2d(-2, -3).norm_infty());
    }

    #[test]
    fn test_index() {
        let v = Vec2d::new(3, 7);
        assert_eq!(3, v[0]);
        assert_eq!(7, v[1]);
    }

    #[test]
    fn test_partial_ord_none() {
        let u = Vec2d::new(2, 1);
        let v = Vec2d::new(3, -7);
        assert_eq!(None, u.partial_cmp(&v));
    }
    #[test]
    fn test_partial_ord_less() {
        let u = Vec2d::new(2, 1);
        let v = Vec2d::new(3, 7);
        assert_eq!(Some(Ordering::Less), u.partial_cmp(&v));
    }
    #[test]
    fn test_partial_ord_equal() {
        let v = Vec2d::new(3, 7);
        assert_eq!(Some(Ordering::Equal), v.partial_cmp(&v));
    }
    #[test]
    fn test_partial_ord_greater() {
        let u = Vec2d::new(2, 1);
        let v = Vec2d::new(3, 7);
        assert_eq!(Some(Ordering::Greater), v.partial_cmp(&u));
    }

    #[test]
    fn test_add() {
        let u = Vec2d::new(2, 1);
        let v = Vec2d::new(3, 7);
        assert_eq!(v2d(5, 8), u + v);
        assert_eq!(v2d(5, 8), u + &v);
        assert_eq!(v2d(5, 8), &u + v);
        assert_eq!(v2d(5, 8), &u + &v);
    }

    #[test]
    fn test_sub() {
        let u = Vec2d::new(2, 1);
        let v = Vec2d::new(3, 7);
        assert_eq!(v2d(-1, -6), u - v);
        assert_eq!(v2d(-1, -6), u - &v);
        assert_eq!(v2d(-1, -6), &u - v);
        assert_eq!(v2d(-1, -6), &u - &v);
    }

    #[test]
    fn test_neg() {
        let u = Vec2d::new(2, 1);
        assert_eq!(v2d(-2, -1), -u);
        assert_eq!(v2d(-2, -1), -&u);
    }

    #[test]
    fn test_mul_sv() {
        let v = Vec2d::new(3, 7);
        assert_eq!(v2d(6, 14), 2 * v);
        assert_eq!(v2d(6, 14), 2 * &v);
        assert_eq!(v2d(6, 14), &2 * v);
        assert_eq!(v2d(6, 14), &2 * &v);
    }

    #[test]
    fn test_mul_vv() {
        let u = Vec2d::new(2, 1);
        let v = Vec2d::new(3, 7);
        assert_eq!(13, u * v);
        assert_eq!(13, u * &v);
        assert_eq!(13, &u * v);
        assert_eq!(13, &u * &v);
    }

    #[test]
    fn test_add_assign() {
        let mut u = Vec2d::new(2, 1);
        u += Vec2d::new(3, 7);
        assert_eq!(v2d(5, 8), u);
        u += &Vec2d::new(3, 7);
        assert_eq!(v2d(8, 15), u);
    }

    #[test]
    fn test_sub_assign() {
        let mut u = Vec2d::new(2, 1);
        u -= Vec2d::new(3, 7);
        assert_eq!(v2d(-1, -6), u);
        u -= &Vec2d::new(3, 7);
        assert_eq!(v2d(-4, -13), u);
    }

    #[test]
    fn test_min() {
        let u = Vec2d::new(2, 1);
        let v = Vec2d::new(3, 7);
        assert_eq!(v2d(2, 1), u.min(v));
    }

    #[test]
    fn test_max() {
        let u = Vec2d::new(2, 1);
        let v = Vec2d::new(3, 7);
        assert_eq!(v2d(3, 7), u.max(v));
    }
}
