//! 2-dimensional vectors.

use core::cmp::Ordering;
use core::fmt;
use core::iter;
use core::ops::Add;
use core::ops::AddAssign;
use core::ops::Div;
use core::ops::Index;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Sub;
use core::ops::SubAssign;

use crate::lex_then;
use crate::partial_then;
use crate::scalar_mul;
use crate::Integer;
use crate::Vector;
use crate::VectorOps;

const DIM: usize = 2;

/// A two-dimensional discrete vector.
#[derive(Clone, Copy, Default, Eq, PartialEq, Hash)]
pub struct Vec2d<S = i64>([S; DIM])
where
    S: Integer;

impl<S: Integer> Vec2d<S> {
    /// Creates a new 2d-vector from its coordinates.
    pub fn new(x: S, y: S) -> Vec2d<S> {
        Vec2d([x, y])
    }

    /// Returns the x coordinate of the vector.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v2d;
    /// let v = v2d(2, 3);
    /// assert_eq!(2, v.x());
    /// ```
    pub fn x(&self) -> S {
        self.0[0]
    }
    /// Returns the y coordinate of the vector.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v2d;
    /// let v = v2d(2, 3);
    /// assert_eq!(3, v.y());
    /// ```
    pub fn y(&self) -> S {
        self.0[1]
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
    /// # use lowdim::v2d;
    /// let v = v2d(3, 2);
    /// assert!(v.is_towards_pos_x());
    /// let v = v2d(3, -2);
    /// assert!(v.is_towards_pos_x());
    /// let v = v2d(-3, -2);
    /// assert!(!v.is_towards_pos_x());
    /// let v = v2d(3, 3);
    /// assert!(!v.is_towards_pos_x());
    /// ```
    pub fn is_towards_pos_x(&self) -> bool {
        self.x() > self.y().abs()
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
    /// # use lowdim::v2d;
    /// let v = v2d(-3, 2);
    /// assert!(v.is_towards_neg_x());
    /// ```
    pub fn is_towards_neg_x(&self) -> bool {
        -self.x() > self.y().abs()
    }
    /// Returns `true` if the vector points towards positive y.
    ///
    /// See [`Vec2d::is_towards_pos_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v2d;
    /// let v = v2d(2, 3);
    /// assert!(v.is_towards_pos_y());
    /// ```
    pub fn is_towards_pos_y(self) -> bool {
        self.y() > self.x().abs()
    }
    /// Returns `true` if the vector points towards negative y.
    ///
    /// See [`Vec2d::is_towards_neg_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v2d;
    /// let v = v2d(2, -3);
    /// assert!(v.is_towards_neg_y());
    /// ```
    pub fn is_towards_neg_y(&self) -> bool {
        -self.y() > self.x().abs()
    }

    /// Returns a vector obtained by rotating this vector left by a right angle.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v2d;
    /// let v = v2d(2, 3);
    /// assert_eq!(v2d(-3, 2), v.rotate_left());
    /// ```
    pub fn rotate_left(&self) -> Self {
        v2d(-self.y(), self.x())
    }
    /// Returns a vector obtained by rotating this vector right by a right angle.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v2d;
    /// let v = v2d(2, 3);
    /// assert_eq!(v2d(3, -2), v.rotate_right());
    /// ```
    pub fn rotate_right(&self) -> Self {
        v2d(self.y(), -self.x())
    }
}

impl<S: Integer> VectorOps<S, Vec2d<S>> for Vec2d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec2d<S>> for Vec2d<S> {}
impl<'a, S: Integer> VectorOps<S, Vec2d<S>, Vec2d<S>> for &'a Vec2d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec2d<S>, Vec2d<S>> for &'a Vec2d<S> {}

impl<S: Integer> Vector<S> for Vec2d<S> {
    const DIM: usize = 2;

    fn with<F>(f: F) -> Vec2d<S>
    where
        F: Fn(usize) -> S,
    {
        Vec2d([f(0), f(1)])
    }

    /// Returns a slice containing the coordinates of the vector.
    fn as_slice(&self) -> &[S] {
        &self.0
    }

    /// Returns a mutable slice containing the coordinates of the vector.
    fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.0
    }

    /// Returns the L1 norm of the vector.
    ///
    /// This is also called the taxicab, Manhatten or city block norm.
    fn norm_l1(&self) -> S {
        let abs_x = self.x().abs();
        let abs_y = self.y().abs();
        abs_x + abs_y
    }
    /// Returns the L∞ norm of the vector.
    ///
    /// This is also called the maximum or Chebychev norm.
    fn norm_l_infty(&self) -> S {
        let abs_x = self.x().abs();
        let abs_y = self.y().abs();
        abs_x.max(abs_y)
    }
    /// Returns the square of the L2-norm of the vector.
    ///
    /// The L2-norm is also called the Euclidean norm and
    /// is the standard notion of the length of a vector.
    fn norm_l2_squared(&self) -> S {
        self * self
    }

    /// Creates a vector of the 8 vectors with L∞ norm equal to 1.
    fn unit_vecs_l_infty() -> Vec<Self> {
        let mut result = Vec::new();
        let zero = S::zero();
        let one = S::one();
        for y in [-one, zero, one] {
            for x in [-one, zero, one] {
                if x != zero || y != zero {
                    result.push(v2d(x, y));
                }
            }
        }
        result
    }

    fn componentwise_cmp(&self, other: &Vec2d<S>) -> Option<Ordering> {
        let x_ordering = Some(self.x().cmp(&other.x()));
        let y_ordering = Some(self.y().cmp(&other.y()));
        partial_then(x_ordering, y_ordering)
    }

    fn lex_cmp(&self, other: &Vec2d<S>) -> Ordering {
        let x_ordering = self.x().cmp(&other.x());
        let y_ordering = self.y().cmp(&other.y());
        lex_then(x_ordering, y_ordering)
    }
}

/// Creates a 2d-vector.
///
/// This is a utility function for concisely representing vectors.
pub fn v2d<S: Integer>(x: S, y: S) -> Vec2d<S> {
    Vec2d::new(x, y)
}

impl<S: Integer> iter::FromIterator<S> for Vec2d<S> {
    fn from_iter<II>(ii: II) -> Self
    where
        II: IntoIterator<Item = S>,
    {
        let mut i = ii.into_iter();
        let x = i.next().unwrap();
        let y = i.next().unwrap();
        Vec2d([x, y])
    }
}

impl<S: Integer> fmt::Debug for Vec2d<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        let mut sep = "";
        for i in 0..Self::DIM {
            write!(f, "{}{}", sep, self[i])?;
            sep = ", ";
        }
        write!(f, ")")
    }
}

impl<S: Integer> Index<usize> for Vec2d<S> {
    type Output = S;
    fn index(&self, i: usize) -> &S {
        self.0.index(i)
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

scalar_mul!(i32, Vec2d<i32>);
scalar_mul!(i64, Vec2d<i64>);
scalar_mul!(i128, Vec2d<i128>);

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

impl<S: Integer> Div<S> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn div(self, other: S) -> Vec2d<S> {
        Vec2d::with(|i| self[i] / other)
    }
}
impl<'a, S: Integer> Div<&'a S> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn div(self, other: &'a S) -> Vec2d<S> {
        Vec2d::with(|i| self[i] / other)
    }
}
impl<'a, S: Integer> Div<S> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn div(self, other: S) -> Vec2d<S> {
        Vec2d::with(|i| self[i] / other)
    }
}
impl<'a, S: Integer> Div<&'a S> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn div(self, other: &'a S) -> Vec2d<S> {
        Vec2d::with(|i| self[i] / other)
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
        let v = v2d(3, 7);
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
    fn test_as_slice() {
        assert_eq!(&[2, 3], v2d(2, 3).as_slice());
    }
    #[test]
    fn test_as_mut_slice() {
        let mut v = v2d(2, 3);
        let s = v.as_mut_slice();
        s[1] += 1;
        assert_eq!(&[2, 4], s);
    }

    #[test]
    fn test_signum() {
        assert_eq!(v2d(1, 1), v2d(2, 3).signum());
        assert_eq!(v2d(1, 0), v2d(2, 0).signum());
        assert_eq!(v2d(1, -1), v2d(2, -3).signum());
        assert_eq!(v2d(0, 1), v2d(0, 3).signum());
        assert_eq!(v2d(0, 0), v2d(0, 0).signum());
        assert_eq!(v2d(0, -1), v2d(0, -3).signum());
        assert_eq!(v2d(-1, 1), v2d(-2, 3).signum());
        assert_eq!(v2d(-1, 0), v2d(-2, 0).signum());
        assert_eq!(v2d(-1, -1), v2d(-2, -3).signum());
    }

    #[test]
    fn test_unit_vecs() {
        let mut uv: Vec<Vec2d<i64>> = Vec2d::unit_vecs();
        uv.sort_by(Vec2d::lex_cmp);
        assert_eq!(vec![v2d(0, 1), v2d(1, 0)], uv);
    }

    #[test]
    fn test_unit_vecs_l1() {
        let mut uv: Vec<Vec2d<i64>> = Vec2d::unit_vecs_l1();
        uv.sort_by(Vec2d::lex_cmp);
        assert_eq!(vec![v2d(-1, 0), v2d(0, -1), v2d(0, 1), v2d(1, 0)], uv);
    }

    #[test]
    fn test_unit_vecs_l_infty() {
        let mut uv: Vec<Vec2d<i64>> = Vec2d::unit_vecs_l_infty();
        uv.sort_by(Vec2d::lex_cmp);
        assert_eq!(
            vec![
                v2d(-1, -1),
                v2d(-1, 0),
                v2d(-1, 1),
                v2d(0, -1),
                v2d(0, 1),
                v2d(1, -1),
                v2d(1, 0),
                v2d(1, 1)
            ],
            uv
        );
    }

    #[test]
    fn test_norm_l1() {
        assert_eq!(5, v2d(2, 3).norm_l1());
        assert_eq!(5, v2d(-2, 3).norm_l1());
        assert_eq!(5, v2d(2, -3).norm_l1());
        assert_eq!(5, v2d(-2, -3).norm_l1());
    }
    #[test]
    fn test_norm_l_infty() {
        assert_eq!(3, v2d(2, 3).norm_l_infty());
        assert_eq!(3, v2d(-2, 3).norm_l_infty());
        assert_eq!(3, v2d(2, -3).norm_l_infty());
        assert_eq!(3, v2d(-2, -3).norm_l_infty());
    }
    #[test]
    fn test_norm_l2_squared() {
        assert_eq!(13, v2d(2, 3).norm_l2_squared());
        assert_eq!(13, v2d(-2, 3).norm_l2_squared());
        assert_eq!(13, v2d(2, -3).norm_l2_squared());
        assert_eq!(13, v2d(-2, -3).norm_l2_squared());
    }

    #[test]
    fn test_from_iter() {
        assert_eq!(v2d(1, 2), vec![1, 2].into_iter().collect::<Vec2d>());
    }

    #[test]
    fn test_debug() {
        assert_eq!("(2, -3)", format!("{:?}", v2d(2, -3)));
    }

    #[test]
    fn test_index() {
        let v = v2d(3, 7);
        assert_eq!(3, v[0]);
        assert_eq!(7, v[1]);
    }

    #[test]
    fn test_componentwise_cmp_none() {
        let u = v2d(2, 1);
        let v = v2d(3, -7);
        assert_eq!(None, u.componentwise_cmp(&v));
    }
    #[test]
    fn test_componentwise_cmp_less() {
        let u = v2d(2, 1);
        let v = v2d(3, 7);
        assert_eq!(Some(Ordering::Less), u.componentwise_cmp(&v));
    }
    #[test]
    fn test_componentwise_cmp_equal() {
        let v = v2d(3, 7);
        assert_eq!(Some(Ordering::Equal), v.componentwise_cmp(&v));
    }
    #[test]
    fn test_componentwise_cmp_greater() {
        let u = v2d(2, 1);
        let v = v2d(3, 7);
        assert_eq!(Some(Ordering::Greater), v.componentwise_cmp(&u));
    }

    #[test]
    fn test_add() {
        let u = v2d(2, 1);
        let v = v2d(3, 7);
        assert_eq!(v2d(5, 8), u + v);
        assert_eq!(v2d(5, 8), u + &v);
        assert_eq!(v2d(5, 8), &u + v);
        assert_eq!(v2d(5, 8), &u + &v);
    }

    #[test]
    fn test_sub() {
        let u = v2d(2, 1);
        let v = v2d(3, 7);
        assert_eq!(v2d(-1, -6), u - v);
        assert_eq!(v2d(-1, -6), u - &v);
        assert_eq!(v2d(-1, -6), &u - v);
        assert_eq!(v2d(-1, -6), &u - &v);
    }

    #[test]
    fn test_neg() {
        let u = v2d(2, 1);
        assert_eq!(v2d(-2, -1), -u);
        assert_eq!(v2d(-2, -1), -&u);
    }

    #[test]
    fn test_mul_sv_32() {
        let v: Vec2d<i32> = v2d(3, 7);
        assert_eq!(v2d(6, 14), 2 * v);
        assert_eq!(v2d(6, 14), 2 * &v);
        assert_eq!(v2d(6, 14), &2 * v);
        assert_eq!(v2d(6, 14), &2 * &v);
    }
    #[test]
    fn test_mul_sv_64() {
        let v: Vec2d<i64> = v2d(3, 7);
        assert_eq!(v2d(6, 14), 2 * v);
        assert_eq!(v2d(6, 14), 2 * &v);
        assert_eq!(v2d(6, 14), &2 * v);
        assert_eq!(v2d(6, 14), &2 * &v);
    }

    #[test]
    fn test_mul_vv() {
        let u = v2d(2, 1);
        let v = v2d(3, 7);
        assert_eq!(13, u * v);
        assert_eq!(13, u * &v);
        assert_eq!(13, &u * v);
        assert_eq!(13, &u * &v);
    }

    #[test]
    fn test_div_vs() {
        let v = v2d(6, 14);
        assert_eq!(v2d(3, 7), v / 2);
        assert_eq!(v2d(3, 7), &v / 2);
        assert_eq!(v2d(3, 7), v / &2);
        assert_eq!(v2d(3, 7), &v / &2);
    }

    #[test]
    fn test_add_assign() {
        let mut u = v2d(2, 1);
        u += v2d(3, 7);
        assert_eq!(v2d(5, 8), u);
        u += &v2d(3, 7);
        assert_eq!(v2d(8, 15), u);
    }

    #[test]
    fn test_sub_assign() {
        let mut u = v2d(2, 1);
        u -= v2d(3, 7);
        assert_eq!(v2d(-1, -6), u);
        u -= &v2d(3, 7);
        assert_eq!(v2d(-4, -13), u);
    }

    #[test]
    fn test_min() {
        let u = v2d(2, 1);
        let v = v2d(3, 7);
        assert_eq!(v2d(2, 1), u.min(v));
    }

    #[test]
    fn test_max() {
        let u = v2d(2, 1);
        let v = v2d(3, 7);
        assert_eq!(v2d(3, 7), u.max(v));
    }

    #[test]
    fn test_is_zero_true() {
        let v = v2d(0, 0);
        assert!(v.is_zero());
    }
    #[test]
    fn test_is_zero_false_x() {
        let v = v2d(1, 0);
        assert!(!v.is_zero());
    }
    #[test]
    fn test_is_zero_false_y() {
        let v = v2d(0, 1);
        assert!(!v.is_zero());
    }
}
