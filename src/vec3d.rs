//! 3-dimensional vectors.

#![warn(missing_docs)]

use core::cmp::Ordering;
use core::ops::Add;
use core::ops::AddAssign;
use core::ops::Index;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Sub;
use core::ops::SubAssign;

use crate::lex_then;
use crate::partial_then;
use crate::Integer;
use crate::Vector;
use crate::VectorOps;

const DIM: usize = 3;

/// A three-dimensional discrete vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Vec3d<S = i64>([S; DIM])
where
    S: Integer;

impl<S: Integer> Vec3d<S> {
    /// Creates a new 3d-vector from its coordinates.
    pub fn new(x: S, y: S, z: S) -> Vec3d<S> {
        Vec3d([x, y, z])
    }

    /// Returns the x coordinate of the vector.
    ///
    /// # Examples
    /// ```
    /// # use gamedim::v3d;
    /// let v = v3d(2, 3, -1);
    /// assert_eq!(2, v.x());
    /// ```
    pub fn x(&self) -> S {
        self.0[0]
    }
    /// Returns the y coordinate of the vector.
    ///
    /// # Examples
    /// ```
    /// # use gamedim::v3d;
    /// let v = v3d(2, 3, -1);
    /// assert_eq!(3, v.y());
    /// ```
    pub fn y(&self) -> S {
        self.0[1]
    }
    /// Returns the z coordinate of the vector.
    ///
    /// # Examples
    /// ```
    /// # use gamedim::v3d;
    /// let v = v3d(2, 3, -1);
    /// assert_eq!(-1, v.z());
    /// ```
    pub fn z(&self) -> S {
        self.0[2]
    }

    /// Returns `true` if the vector points towards positive x.
    ///
    /// That is, among the vectors pointing in both directions along the coordinate axes,
    /// the one pointing towards positive x is closest to this vector.
    /// Or to put it more concretely, the x coordinate of the vector is positive,
    /// and its absolute value is greater than that of the other coordinates.
    ///
    /// # Examples
    /// ```
    /// use gamedim::v3d;
    ///
    /// let v = v3d(3, 2, -1);
    /// assert!(v.is_towards_pos_x());
    /// let v = v3d(3, -2, -1);
    /// assert!(v.is_towards_pos_x());
    /// let v = v3d(-3, -2, -1);
    /// assert!(!v.is_towards_pos_x());
    /// let v = v3d(3, 2, -3);
    /// assert!(!v.is_towards_pos_x());
    /// ```
    pub fn is_towards_pos_x(&self) -> bool {
        self.x() > self.y().abs() && self.x() > self.z().abs()
    }
    /// Returns `true` if the vector points towards negative x.
    ///
    /// That is, among the vectors pointing in both directions along the coordinate axes,
    /// the one pointing towards negative x is closest to this vector.
    /// Or to put it more concretely, the x coordinate of the vector is negative,
    /// and its absolute value is greater than that of the other coordinates.
    ///
    /// # Examples
    /// ```
    /// # use gamedim::v3d;
    /// let v = v3d(-3, 2, -1);
    /// assert!(v.is_towards_neg_x());
    /// ```
    pub fn is_towards_neg_x(&self) -> bool {
        -self.x() > self.y().abs() && -self.x() > self.z().abs()
    }
    /// Returns `true` if the vector points towards positive y.
    ///
    /// See [`Vec3d::is_towards_pos_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use gamedim::v3d;
    /// let v = v3d(2, 3, -1);
    /// assert!(v.is_towards_pos_y());
    /// ```
    pub fn is_towards_pos_y(self) -> bool {
        self.y() > self.x().abs() && self.y() > self.z().abs()
    }
    /// Returns `true` if the vector points towards negative y.
    ///
    /// See [`Vec3d::is_towards_neg_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use gamedim::v3d;
    /// let v = v3d(2, -3, -1);
    /// assert!(v.is_towards_neg_y());
    /// ```
    pub fn is_towards_neg_y(&self) -> bool {
        -self.y() > self.x().abs() && -self.y() > self.z().abs()
    }
    /// Returns `true` if the vector points towards positive z.
    ///
    /// See [`Vec3d::is_towards_pos_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use gamedim::v3d;
    /// let v = v3d(2, -3, 4);
    /// assert!(v.is_towards_pos_z());
    /// ```
    pub fn is_towards_pos_z(self) -> bool {
        self.z() > self.x().abs() && self.z() > self.y().abs()
    }
    /// Returns `true` if the vector points towards negative z.
    ///
    /// See [`Vec3d::is_towards_neg_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use gamedim::v3d;
    /// let v = v3d(2, -3, -4);
    /// assert!(v.is_towards_neg_z());
    /// ```
    pub fn is_towards_neg_z(&self) -> bool {
        -self.z() > self.x().abs() && -self.z() > self.y().abs()
    }
}

impl<S: Integer> VectorOps<S, Vec3d<S>> for Vec3d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec3d<S>> for Vec3d<S> {}
impl<'a, S: Integer> VectorOps<S, Vec3d<S>, Vec3d<S>> for &'a Vec3d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec3d<S>, Vec3d<S>> for &'a Vec3d<S> {}

impl<S: Integer> Vector<S> for Vec3d<S> {
    const DIM: usize = 3;

    fn with<F>(f: F) -> Vec3d<S>
    where
        F: Fn(usize) -> S,
    {
        Vec3d([f(0), f(1), f(2)])
    }

    /// The L1, taxicab or Manhatten norm.
    fn norm_l1(&self) -> S {
        let abs_x = self.x().abs();
        let abs_y = self.y().abs();
        let abs_z = self.z().abs();
        abs_x + abs_y + abs_z
    }

    /// The maximum, Chebychev or L-infinity norm.
    fn norm_l_infty(&self) -> S {
        let abs_x = self.x().abs();
        let abs_y = self.y().abs();
        let abs_z = self.z().abs();
        abs_x.max(abs_y).max(abs_z)
    }
    /// Creates a vector of the 26 vectors with L∞ norm equal to 1.
    fn unit_vecs_l_infty() -> Vec<Self> {
        let mut result = Vec::new();
        for z in -1..=1 {
            for y in -1..=1 {
                for x in -1..=1 {
                    if x != 0 || y != 0 || z != 0 {
                        result.push(v3d(S::from(x), S::from(y), S::from(z)));
                    }
                }
            }
        }
        result
    }

    fn lex_cmp(&self, other: &Vec3d<S>) -> Ordering {
        let x_ordering = self.x().cmp(&other.x());
        let y_ordering = self.y().cmp(&other.y());
        let z_ordering = self.z().cmp(&other.z());
        lex_then(lex_then(x_ordering, y_ordering), z_ordering)
    }
}

/// Creates a 3d-vector.
///
/// This is a utility function for concisely representing vectors.
pub fn v3d<S: Integer>(x: S, y: S, z: S) -> Vec3d<S> {
    Vec3d::new(x, y, z)
}

impl<S: Integer> Index<usize> for Vec3d<S> {
    type Output = S;
    fn index(&self, i: usize) -> &S {
        self.0.index(i)
    }
}
impl<S: Integer> PartialOrd for Vec3d<S> {
    fn partial_cmp(&self, other: &Vec3d<S>) -> Option<Ordering> {
        let x_ordering = Some(self.x().cmp(&other.x()));
        let y_ordering = Some(self.y().cmp(&other.y()));
        let z_ordering = Some(self.z().cmp(&other.z()));
        partial_then(partial_then(x_ordering, y_ordering), z_ordering)
    }
}

impl<S: Integer> Add<Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> Sub<Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] - other[i])
    }
}

impl<S: Integer> Neg for Vec3d<S> {
    type Output = Vec3d<S>;

    fn neg(self) -> Vec3d<S> {
        Vec3d::with(|i| -self[i])
    }
}
impl<'a, S: Integer> Neg for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn neg(self) -> Vec3d<S> {
        Vec3d::with(|i| -self[i])
    }
}

impl Mul<Vec3d<i64>> for i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: Vec3d<i64>) -> Vec3d<i64> {
        Vec3d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec3d<i64>> for i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: &'a Vec3d<i64>) -> Vec3d<i64> {
        Vec3d::with(|i| self * other[i])
    }
}

impl<'a> Mul<Vec3d<i64>> for &'a i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: Vec3d<i64>) -> Vec3d<i64> {
        Vec3d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec3d<i64>> for &'a i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: &'a Vec3d<i64>) -> Vec3d<i64> {
        Vec3d::with(|i| self * other[i])
    }
}

impl<S: Integer> Mul<Vec3d<S>> for Vec3d<S> {
    type Output = S;

    fn mul(self, other: Vec3d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl<'a, S: Integer> Mul<&'a Vec3d<S>> for Vec3d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec3d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl<'a, S: Integer> Mul<Vec3d<S>> for &'a Vec3d<S> {
    type Output = S;

    fn mul(self, other: Vec3d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl<'a, S: Integer> Mul<&'a Vec3d<S>> for &'a Vec3d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec3d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl<S: Integer> AddAssign for Vec3d<S> {
    fn add_assign(&mut self, other: Vec3d<S>) {
        *self = Vec3d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> AddAssign<&'a Vec3d<S>> for Vec3d<S> {
    fn add_assign(&mut self, other: &'a Vec3d<S>) {
        *self = Vec3d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> SubAssign for Vec3d<S> {
    fn sub_assign(&mut self, other: Vec3d<S>) {
        *self = Vec3d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> SubAssign<&'a Vec3d<S>> for Vec3d<S> {
    fn sub_assign(&mut self, other: &'a Vec3d<S>) {
        *self = Vec3d::with(|i| self[i] - other[i])
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;
    use core::convert::TryFrom;

    use crate::v3d;
    use crate::Vec3d;
    use crate::Vector;

    #[test]
    fn test_new_x_y() {
        let v = v3d(3, 7, 1);
        assert_eq!(3, v.x());
        assert_eq!(7, v.y());
        assert_eq!(1, v.z());
    }

    #[test]
    fn test_with() {
        assert_eq!(v3d(2, 3, 4), Vec3d::with(|i| i64::try_from(i + 2).unwrap()));
    }

    #[test]
    fn test_unit_vecs() {
        let mut uv: Vec<Vec3d<i64>> = Vec3d::unit_vecs();
        uv.sort_by(Vec3d::lex_cmp);
        assert_eq!(vec![v3d(0, 0, 1), v3d(0, 1, 0), v3d(1, 0, 0)], uv);
    }

    #[test]
    fn test_unit_vecs_l1() {
        let mut uv: Vec<Vec3d<i64>> = Vec3d::unit_vecs_l1();
        uv.sort_by(Vec3d::lex_cmp);
        assert_eq!(
            vec![
                v3d(-1, 0, 0),
                v3d(0, -1, 0),
                v3d(0, 0, -1),
                v3d(0, 0, 1),
                v3d(0, 1, 0),
                v3d(1, 0, 0),
            ],
            uv
        );
    }

    #[test]
    fn test_unit_vecs_l_infty() {
        let mut uv: Vec<Vec3d<i64>> = Vec3d::unit_vecs_l_infty();
        uv.sort_by(Vec3d::lex_cmp);
        assert_eq!(
            vec![
                v3d(-1, -1, -1),
                v3d(-1, -1, 0),
                v3d(-1, -1, 1),
                v3d(-1, 0, -1),
                v3d(-1, 0, 0),
                v3d(-1, 0, 1),
                v3d(-1, 1, -1),
                v3d(-1, 1, 0),
                v3d(-1, 1, 1),
                v3d(0, -1, -1),
                v3d(0, -1, 0),
                v3d(0, -1, 1),
                v3d(0, 0, -1),
                v3d(0, 0, 1),
                v3d(0, 1, -1),
                v3d(0, 1, 0),
                v3d(0, 1, 1),
                v3d(1, -1, -1),
                v3d(1, -1, 0),
                v3d(1, -1, 1),
                v3d(1, 0, -1),
                v3d(1, 0, 0),
                v3d(1, 0, 1),
                v3d(1, 1, -1),
                v3d(1, 1, 0),
                v3d(1, 1, 1),
            ],
            uv
        );
    }

    #[test]
    fn test_norm_l1() {
        assert_eq!(10, v3d(2, 3, 5).norm_l1());
        assert_eq!(10, v3d(-2, 3, 5).norm_l1());
        assert_eq!(10, v3d(2, -3, 5).norm_l1());
        assert_eq!(10, v3d(-2, -3, 5).norm_l1());
        assert_eq!(10, v3d(2, 3, -5).norm_l1());
        assert_eq!(10, v3d(-2, 3, -5).norm_l1());
        assert_eq!(10, v3d(2, -3, -5).norm_l1());
        assert_eq!(10, v3d(-2, -3, -5).norm_l1());
    }
    #[test]
    fn test_norm_l_infty() {
        assert_eq!(5, v3d(2, 3, 5).norm_l_infty());
        assert_eq!(5, v3d(-2, 3, 5).norm_l_infty());
        assert_eq!(5, v3d(2, -3, 5).norm_l_infty());
        assert_eq!(5, v3d(-2, -3, 5).norm_l_infty());
        assert_eq!(5, v3d(2, 3, -5).norm_l_infty());
        assert_eq!(5, v3d(-2, 3, -5).norm_l_infty());
        assert_eq!(5, v3d(2, -3, -5).norm_l_infty());
        assert_eq!(5, v3d(-2, -3, -5).norm_l_infty());
    }

    #[test]
    fn test_index() {
        let v = v3d(3, 7, 1);
        assert_eq!(3, v[0]);
        assert_eq!(7, v[1]);
        assert_eq!(1, v[2]);
    }

    #[test]
    fn test_partial_ord_none() {
        let u = v3d(3, 1, -5);
        let v = v3d(2, 1, 1);
        assert_eq!(None, u.partial_cmp(&v));
    }
    #[test]
    fn test_partial_ord_less() {
        let u = v3d(2, 1, 1);
        let v = v3d(3, 7, 5);
        assert_eq!(Some(Ordering::Less), u.partial_cmp(&v));
    }
    #[test]
    fn test_partial_ord_equal() {
        let v = v3d(3, 7, 5);
        assert_eq!(Some(Ordering::Equal), v.partial_cmp(&v));
    }
    #[test]
    fn test_partial_ord_greater() {
        let u = v3d(2, 1, 1);
        let v = v3d(2, 7, 5);
        assert_eq!(Some(Ordering::Greater), v.partial_cmp(&u));
    }

    #[test]
    fn test_add() {
        let u = v3d(2, 1, 5);
        let v = v3d(3, 7, 1);
        assert_eq!(v3d(5, 8, 6), u + v);
        assert_eq!(v3d(5, 8, 6), u + &v);
        assert_eq!(v3d(5, 8, 6), &u + v);
        assert_eq!(v3d(5, 8, 6), &u + &v);
    }

    #[test]
    fn test_sub() {
        let u = v3d(2, 1, 5);
        let v = v3d(3, 7, 1);
        assert_eq!(v3d(-1, -6, 4), u - v);
        assert_eq!(v3d(-1, -6, 4), u - &v);
        assert_eq!(v3d(-1, -6, 4), &u - v);
        assert_eq!(v3d(-1, -6, 4), &u - &v);
    }

    #[test]
    fn test_neg() {
        let u = v3d(2, 1, -3);
        assert_eq!(v3d(-2, -1, 3), -u);
        assert_eq!(v3d(-2, -1, 3), -&u);
    }

    #[test]
    fn test_mul_sv() {
        let v = v3d(3, 7, 1);
        assert_eq!(v3d(6, 14, 2), 2 * v);
        assert_eq!(v3d(6, 14, 2), 2 * &v);
        assert_eq!(v3d(6, 14, 2), &2 * v);
        assert_eq!(v3d(6, 14, 2), &2 * &v);
    }

    #[test]
    fn test_mul_vv() {
        let u = v3d(2, 1, 5);
        let v = v3d(3, 7, 1);
        assert_eq!(18, u * v);
        assert_eq!(18, u * &v);
        assert_eq!(18, &u * v);
        assert_eq!(18, &u * &v);
    }

    #[test]
    fn test_add_assign() {
        let mut u = v3d(2, 1, 5);
        u += v3d(3, 7, 1);
        assert_eq!(v3d(5, 8, 6), u);
        u += &v3d(3, 7, 1);
        assert_eq!(v3d(8, 15, 7), u);
    }

    #[test]
    fn test_sub_assign() {
        let mut u = v3d(2, 1, 5);
        u -= v3d(3, 7, 1);
        assert_eq!(v3d(-1, -6, 4), u);
        u -= &v3d(3, 7, 1);
        assert_eq!(v3d(-4, -13, 3), u);
    }

    #[test]
    fn test_min() {
        let u = v3d(2, 1, 5);
        let v = v3d(3, 7, 1);
        assert_eq!(v3d(2, 1, 1), u.min(v));
    }

    #[test]
    fn test_max() {
        let u = v3d(2, 1, 5);
        let v = v3d(3, 7, 1);
        assert_eq!(v3d(3, 7, 5), u.max(v));
    }
}
