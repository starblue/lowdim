//! 4d vectors.

use core::cmp::Ordering;
use core::fmt;
use core::iter;
use core::ops::Add;
use core::ops::AddAssign;
use core::ops::Div;
use core::ops::Index;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Rem;
use core::ops::Sub;
use core::ops::SubAssign;

use crate::lex_then;
use crate::partial_then;
use crate::scalar_mul;
use crate::Integer;
use crate::Layout4d;
use crate::Vector;
use crate::VectorOps;

const DIM: usize = 4;

/// A 4d discrete vector.
#[derive(Clone, Copy, Default, Eq, PartialEq, Hash)]
pub struct Vec4d<S = i64>([S; DIM])
where
    S: Integer;

impl<S: Integer> Vec4d<S> {
    /// Creates a new 4d vector from its coordinates.
    pub fn new(x: S, y: S, z: S, w: S) -> Vec4d<S> {
        Vec4d([x, y, z, w])
    }

    /// Returns the x coordinate of the vector.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, 3, -1, 4);
    /// assert_eq!(2, v.x());
    /// ```
    pub fn x(&self) -> S {
        self.0[0]
    }
    /// Returns the y coordinate of the vector.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, 3, -1, 4);
    /// assert_eq!(3, v.y());
    /// ```
    pub fn y(&self) -> S {
        self.0[1]
    }
    /// Returns the z coordinate of the vector.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, 3, -1, 4);
    /// assert_eq!(-1, v.z());
    /// ```
    pub fn z(&self) -> S {
        self.0[2]
    }
    /// Returns the w coordinate of the vector.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, 3, -1, 4);
    /// assert_eq!(4, v.w());
    /// ```
    pub fn w(&self) -> S {
        self.0[3]
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
    /// use lowdim::v4d;
    ///
    /// let v = v4d(6, 2, -1, 4);
    /// assert!(v.is_towards_pos_x());
    /// let v = v4d(6, -2, -1, 4);
    /// assert!(v.is_towards_pos_x());
    /// let v = v4d(-6, -2, -1, 4);
    /// assert!(!v.is_towards_pos_x());
    /// let v = v4d(6, 2, -1, 6);
    /// assert!(!v.is_towards_pos_x());
    /// ```
    pub fn is_towards_pos_x(&self) -> bool {
        self.x() > self.y().abs() && self.x() > self.z().abs() && self.x() > self.w().abs()
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
    /// # use lowdim::v4d;
    /// let v = v4d(-6, 2, -1, 4);
    /// assert!(v.is_towards_neg_x());
    /// ```
    pub fn is_towards_neg_x(&self) -> bool {
        -self.x() > self.y().abs() && -self.x() > self.z().abs() && -self.x() > self.w().abs()
    }
    /// Returns `true` if the vector points towards positive y.
    ///
    /// See [`Vec4d::is_towards_pos_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, 6, -1, 4);
    /// assert!(v.is_towards_pos_y());
    /// ```
    pub fn is_towards_pos_y(self) -> bool {
        self.y() > self.x().abs() && self.y() > self.z().abs() && self.y() > self.w().abs()
    }
    /// Returns `true` if the vector points towards negative y.
    ///
    /// See [`Vec4d::is_towards_neg_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, -6, -1, 4);
    /// assert!(v.is_towards_neg_y());
    /// ```
    pub fn is_towards_neg_y(&self) -> bool {
        -self.y() > self.x().abs() && -self.y() > self.z().abs() && -self.y() > self.w().abs()
    }
    /// Returns `true` if the vector points towards positive z.
    ///
    /// See [`Vec4d::is_towards_pos_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, -3, 4, -1);
    /// assert!(v.is_towards_pos_z());
    /// ```
    pub fn is_towards_pos_z(self) -> bool {
        self.z() > self.x().abs() && self.z() > self.y().abs() && self.z() > self.w().abs()
    }
    /// Returns `true` if the vector points towards negative z.
    ///
    /// See [`Vec4d::is_towards_neg_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, -3, -4, -1);
    /// assert!(v.is_towards_neg_z());
    /// ```
    pub fn is_towards_neg_z(&self) -> bool {
        -self.z() > self.x().abs() && -self.z() > self.y().abs() && -self.z() > self.w().abs()
    }
    /// Returns `true` if the vector points towards positive w.
    ///
    /// See [`Vec4d::is_towards_pos_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, -3, 4, 5);
    /// assert!(v.is_towards_pos_w());
    /// ```
    pub fn is_towards_pos_w(self) -> bool {
        self.w() > self.x().abs() && self.w() > self.y().abs() && self.w() > self.z().abs()
    }
    /// Returns `true` if the vector points towards negative w.
    ///
    /// See [`Vec4d::is_towards_neg_x`] for more details.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::v4d;
    /// let v = v4d(2, -3, -4, -5);
    /// assert!(v.is_towards_neg_w());
    /// ```
    pub fn is_towards_neg_w(&self) -> bool {
        -self.w() > self.x().abs() && -self.w() > self.y().abs() && -self.w() > self.z().abs()
    }
}

impl<S: Integer> VectorOps<S, Vec4d<S>> for Vec4d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec4d<S>> for Vec4d<S> {}
impl<'a, S: Integer> VectorOps<S, Vec4d<S>, Vec4d<S>> for &'a Vec4d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec4d<S>, Vec4d<S>> for &'a Vec4d<S> {}

impl<S: Integer> Vector<S> for Vec4d<S> {
    const DIM: usize = 4;

    type DefaultLayout = Layout4d<S>;

    fn with<F>(f: F) -> Vec4d<S>
    where
        F: Fn(usize) -> S,
    {
        Vec4d([f(0), f(1), f(2), f(3)])
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
        let abs_z = self.z().abs();
        let abs_w = self.w().abs();
        abs_x + abs_y + abs_z + abs_w
    }
    /// Returns the L∞ norm of the vector.
    ///
    /// This is also called the maximum or Chebychev norm.
    fn norm_l_infty(&self) -> S {
        let abs_x = self.x().abs();
        let abs_y = self.y().abs();
        let abs_z = self.z().abs();
        let abs_w = self.w().abs();
        abs_x.max(abs_y).max(abs_z).max(abs_w)
    }
    /// Returns the square of the L2-norm of the vector.
    ///
    /// The L2-norm is also called the Euclidean norm and
    /// is the standard notion of the length of a vector.
    fn norm_l2_squared(&self) -> S {
        self * self
    }

    fn componentwise_cmp(&self, other: &Vec4d<S>) -> Option<Ordering> {
        let x_ordering = Some(self.x().cmp(&other.x()));
        let y_ordering = Some(self.y().cmp(&other.y()));
        let z_ordering = Some(self.z().cmp(&other.z()));
        let w_ordering = Some(self.w().cmp(&other.w()));
        partial_then(
            partial_then(partial_then(x_ordering, y_ordering), z_ordering),
            w_ordering,
        )
    }

    fn lex_cmp(&self, other: &Vec4d<S>) -> Ordering {
        let x_ordering = self.x().cmp(&other.x());
        let y_ordering = self.y().cmp(&other.y());
        let z_ordering = self.z().cmp(&other.z());
        let w_ordering = self.w().cmp(&other.w());
        lex_then(
            lex_then(lex_then(x_ordering, y_ordering), z_ordering),
            w_ordering,
        )
    }
}

/// Creates a 4d vector.
///
/// This is a utility function for concisely representing vectors.
pub fn v4d<S: Integer>(x: S, y: S, z: S, w: S) -> Vec4d<S> {
    Vec4d::new(x, y, z, w)
}

impl<S: Integer> iter::FromIterator<S> for Vec4d<S> {
    fn from_iter<II>(ii: II) -> Self
    where
        II: IntoIterator<Item = S>,
    {
        let mut i = ii.into_iter();
        let x = i.next().unwrap();
        let y = i.next().unwrap();
        let z = i.next().unwrap();
        let w = i.next().unwrap();
        Vec4d([x, y, z, w])
    }
}

impl<S: Integer> iter::Sum<Vec4d<S>> for Vec4d<S> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Vec4d<S>>,
    {
        iter.fold(Vec4d::zero(), |sum, v| sum + v)
    }
}
impl<'a, S: Integer> iter::Sum<&'a Vec4d<S>> for Vec4d<S> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Vec4d<S>>,
    {
        iter.fold(Vec4d::zero(), |sum, v| sum + v)
    }
}

impl<S: Integer> fmt::Debug for Vec4d<S> {
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

scalar_mul!(i32, Vec4d<i32>);
scalar_mul!(i64, Vec4d<i64>);
scalar_mul!(i128, Vec4d<i128>);

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

impl<S: Integer> Div<S> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn div(self, other: S) -> Vec4d<S> {
        Vec4d::with(|i| self[i].div_euclid(other))
    }
}
impl<'a, S: Integer> Div<&'a S> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn div(self, other: &'a S) -> Vec4d<S> {
        Vec4d::with(|i| self[i].div_euclid(*other))
    }
}
impl<'a, S: Integer> Div<S> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn div(self, other: S) -> Vec4d<S> {
        Vec4d::with(|i| self[i].div_euclid(other))
    }
}
impl<'a, S: Integer> Div<&'a S> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn div(self, other: &'a S) -> Vec4d<S> {
        Vec4d::with(|i| self[i].div_euclid(*other))
    }
}

impl<S: Integer> Rem<S> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn rem(self, other: S) -> Vec4d<S> {
        Vec4d::with(|i| self[i].rem_euclid(other))
    }
}
impl<'a, S: Integer> Rem<&'a S> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn rem(self, other: &'a S) -> Vec4d<S> {
        Vec4d::with(|i| self[i].rem_euclid(*other))
    }
}
impl<'a, S: Integer> Rem<S> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn rem(self, other: S) -> Vec4d<S> {
        Vec4d::with(|i| self[i].rem_euclid(other))
    }
}
impl<'a, S: Integer> Rem<&'a S> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn rem(self, other: &'a S) -> Vec4d<S> {
        Vec4d::with(|i| self[i].rem_euclid(*other))
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
    use core::cmp::Ordering;
    use core::convert::TryFrom;

    use crate::v4d;
    use crate::Vec4d;
    use crate::Vector;

    #[test]
    fn test_new_x_y() {
        let v = v4d(3, 7, 1, -2);
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
    fn test_as_slice() {
        assert_eq!(&[2, 3, 4, 5], v4d(2, 3, 4, 5).as_slice());
    }
    #[test]
    fn test_as_mut_slice() {
        let mut v = v4d(2, 3, 4, 5);
        let s = v.as_mut_slice();
        s[1] += 1;
        assert_eq!(&[2, 4, 4, 5], s);
    }

    #[test]
    fn test_unit_vecs() {
        let mut uv = Vec4d::unit_vecs().collect::<Vec<Vec4d<i64>>>();
        uv.sort_by(Vec4d::lex_cmp);
        assert_eq!(
            vec![
                v4d(0, 0, 0, 1),
                v4d(0, 0, 1, 0),
                v4d(0, 1, 0, 0),
                v4d(1, 0, 0, 0)
            ],
            uv
        );
    }

    #[test]
    fn test_unit_vecs_l1() {
        let mut uv = Vec4d::unit_vecs_l1().collect::<Vec<Vec4d<i64>>>();
        uv.sort_by(Vec4d::lex_cmp);
        assert_eq!(
            vec![
                v4d(-1, 0, 0, 0),
                v4d(0, -1, 0, 0),
                v4d(0, 0, -1, 0),
                v4d(0, 0, 0, -1),
                v4d(0, 0, 0, 1),
                v4d(0, 0, 1, 0),
                v4d(0, 1, 0, 0),
                v4d(1, 0, 0, 0),
            ],
            uv
        );
    }

    #[test]
    fn test_unit_vecs_l_infty() {
        let mut uv = Vec4d::unit_vecs_l_infty().collect::<Vec<Vec4d<i64>>>();
        uv.sort_by(Vec4d::lex_cmp);
        assert_eq!(
            vec![
                v4d(-1, -1, -1, -1),
                v4d(-1, -1, -1, 0),
                v4d(-1, -1, -1, 1),
                v4d(-1, -1, 0, -1),
                v4d(-1, -1, 0, 0),
                v4d(-1, -1, 0, 1),
                v4d(-1, -1, 1, -1),
                v4d(-1, -1, 1, 0),
                v4d(-1, -1, 1, 1),
                v4d(-1, 0, -1, -1),
                v4d(-1, 0, -1, 0),
                v4d(-1, 0, -1, 1),
                v4d(-1, 0, 0, -1),
                v4d(-1, 0, 0, 0),
                v4d(-1, 0, 0, 1),
                v4d(-1, 0, 1, -1),
                v4d(-1, 0, 1, 0),
                v4d(-1, 0, 1, 1),
                v4d(-1, 1, -1, -1),
                v4d(-1, 1, -1, 0),
                v4d(-1, 1, -1, 1),
                v4d(-1, 1, 0, -1),
                v4d(-1, 1, 0, 0),
                v4d(-1, 1, 0, 1),
                v4d(-1, 1, 1, -1),
                v4d(-1, 1, 1, 0),
                v4d(-1, 1, 1, 1),
                v4d(0, -1, -1, -1),
                v4d(0, -1, -1, 0),
                v4d(0, -1, -1, 1),
                v4d(0, -1, 0, -1),
                v4d(0, -1, 0, 0),
                v4d(0, -1, 0, 1),
                v4d(0, -1, 1, -1),
                v4d(0, -1, 1, 0),
                v4d(0, -1, 1, 1),
                v4d(0, 0, -1, -1),
                v4d(0, 0, -1, 0),
                v4d(0, 0, -1, 1),
                v4d(0, 0, 0, -1),
                v4d(0, 0, 0, 1),
                v4d(0, 0, 1, -1),
                v4d(0, 0, 1, 0),
                v4d(0, 0, 1, 1),
                v4d(0, 1, -1, -1),
                v4d(0, 1, -1, 0),
                v4d(0, 1, -1, 1),
                v4d(0, 1, 0, -1),
                v4d(0, 1, 0, 0),
                v4d(0, 1, 0, 1),
                v4d(0, 1, 1, -1),
                v4d(0, 1, 1, 0),
                v4d(0, 1, 1, 1),
                v4d(1, -1, -1, -1),
                v4d(1, -1, -1, 0),
                v4d(1, -1, -1, 1),
                v4d(1, -1, 0, -1),
                v4d(1, -1, 0, 0),
                v4d(1, -1, 0, 1),
                v4d(1, -1, 1, -1),
                v4d(1, -1, 1, 0),
                v4d(1, -1, 1, 1),
                v4d(1, 0, -1, -1),
                v4d(1, 0, -1, 0),
                v4d(1, 0, -1, 1),
                v4d(1, 0, 0, -1),
                v4d(1, 0, 0, 0),
                v4d(1, 0, 0, 1),
                v4d(1, 0, 1, -1),
                v4d(1, 0, 1, 0),
                v4d(1, 0, 1, 1),
                v4d(1, 1, -1, -1),
                v4d(1, 1, -1, 0),
                v4d(1, 1, -1, 1),
                v4d(1, 1, 0, -1),
                v4d(1, 1, 0, 0),
                v4d(1, 1, 0, 1),
                v4d(1, 1, 1, -1),
                v4d(1, 1, 1, 0),
                v4d(1, 1, 1, 1),
            ],
            uv
        );
    }

    #[test]
    fn test_norm_l1() {
        assert_eq!(18, v4d(2, 3, 5, 8).norm_l1());
        assert_eq!(18, v4d(-2, -3, -5, -8).norm_l1());
    }
    #[test]
    fn test_norm_l_infty() {
        assert_eq!(8, v4d(2, 3, 5, 8).norm_l_infty());
        assert_eq!(8, v4d(-2, -3, -5, -8).norm_l_infty());
    }
    #[test]
    fn test_norm_l2_squared() {
        assert_eq!(102, v4d(2, 3, 5, 8).norm_l2_squared());
        assert_eq!(102, v4d(-2, -3, -5, -8).norm_l2_squared());
    }

    #[test]
    fn test_from_iter() {
        assert_eq!(
            v4d(1, 2, 3, 4),
            vec![1, 2, 3, 4].into_iter().collect::<Vec4d>()
        );
    }

    #[test]
    fn test_sum() {
        let vs = vec![v4d(1, 0, -1, -2), v4d(0, 1, 2, 3), v4d(2, -3, 1, 0)];
        assert_eq!(v4d(3, -2, 2, 1), vs.iter().map(|v: &Vec4d| v).sum());
        assert_eq!(v4d(3, -2, 2, 1), vs.into_iter().map(|v: Vec4d| v).sum());
    }

    #[test]
    fn test_debug() {
        assert_eq!("(2, -3, 5, -8)", format!("{:?}", v4d(2, -3, 5, -8)));
    }

    #[test]
    fn test_index() {
        let v = v4d(3, 7, 1, -2);
        assert_eq!(3, v[0]);
        assert_eq!(7, v[1]);
        assert_eq!(1, v[2]);
        assert_eq!(-2, v[3]);
    }

    #[test]
    fn test_componentwise_cmp_none() {
        let u = v4d(2, 1, 5, -4);
        let v = v4d(2, 7, 1, 4);
        assert_eq!(None, u.componentwise_cmp(&v));
    }
    #[test]
    fn test_componentwise_cmp_less() {
        let u = v4d(2, 1, 1, -4);
        let v = v4d(3, 1, 5, 4);
        assert_eq!(Some(Ordering::Less), u.componentwise_cmp(&v));
    }
    #[test]
    fn test_componentwise_cmp_equal() {
        let v = v4d(3, 7, 5, 4);
        assert_eq!(Some(Ordering::Equal), v.componentwise_cmp(&v));
    }
    #[test]
    fn test_componentwise_cmp_greater() {
        let u = v4d(2, 1, 1, -4);
        let v = v4d(3, 1, 5, 4);
        assert_eq!(Some(Ordering::Greater), v.componentwise_cmp(&u));
    }

    #[test]
    fn test_add() {
        let u = v4d(2, 1, 5, -4);
        let v = v4d(3, 7, 1, 4);
        assert_eq!(v4d(5, 8, 6, 0), u + v);
        assert_eq!(v4d(5, 8, 6, 0), u + &v);
        assert_eq!(v4d(5, 8, 6, 0), &u + v);
        assert_eq!(v4d(5, 8, 6, 0), &u + &v);
    }

    #[test]
    fn test_sub() {
        let u = v4d(2, 1, 5, -4);
        let v = v4d(3, 7, 1, 4);
        assert_eq!(v4d(-1, -6, 4, -8), u - v);
        assert_eq!(v4d(-1, -6, 4, -8), u - &v);
        assert_eq!(v4d(-1, -6, 4, -8), &u - v);
        assert_eq!(v4d(-1, -6, 4, -8), &u - &v);
    }

    #[test]
    fn test_neg() {
        let u = v4d(2, 1, -3, 7);
        assert_eq!(v4d(-2, -1, 3, -7), -u);
        assert_eq!(v4d(-2, -1, 3, -7), -&u);
    }

    #[test]
    fn test_mul_sv_32() {
        let v: Vec4d<i32> = v4d(3, 7, 1, 4);
        assert_eq!(v4d(6, 14, 2, 8), 2 * v);
        assert_eq!(v4d(6, 14, 2, 8), 2 * &v);
        assert_eq!(v4d(6, 14, 2, 8), &2 * v);
        assert_eq!(v4d(6, 14, 2, 8), &2 * &v);
    }
    #[test]
    fn test_mul_sv_64() {
        let v: Vec4d<i64> = v4d(3, 7, 1, 4);
        assert_eq!(v4d(6, 14, 2, 8), 2 * v);
        assert_eq!(v4d(6, 14, 2, 8), 2 * &v);
        assert_eq!(v4d(6, 14, 2, 8), &2 * v);
        assert_eq!(v4d(6, 14, 2, 8), &2 * &v);
    }

    #[test]
    fn test_mul_vv() {
        let u = v4d(2, 1, 5, -4);
        let v = v4d(3, 7, 1, 4);
        assert_eq!(2, u * v);
        assert_eq!(2, u * &v);
        assert_eq!(2, &u * v);
        assert_eq!(2, &u * &v);
    }

    #[test]
    fn test_div_vs() {
        let v = v4d(-5, 14, 2, 8);
        assert_eq!(v4d(-2, 4, 0, 2), v / 3);
        assert_eq!(v4d(-2, 4, 0, 2), &v / 3);
        assert_eq!(v4d(-2, 4, 0, 2), v / &3);
        assert_eq!(v4d(-2, 4, 0, 2), &v / &3);
    }

    #[test]
    fn test_rem_vs() {
        let v = v4d(-5, 14, 2, 8);
        assert_eq!(v4d(1, 2, 2, 2), v % 3);
        assert_eq!(v4d(1, 2, 2, 2), &v % 3);
        assert_eq!(v4d(1, 2, 2, 2), v % &3);
        assert_eq!(v4d(1, 2, 2, 2), &v % &3);
    }

    #[test]
    fn test_add_assign() {
        let mut u = v4d(2, 1, 5, 4);
        u += v4d(3, 7, 1, -4);
        assert_eq!(v4d(5, 8, 6, 0), u);
        u += &v4d(3, 7, 1, -4);
        assert_eq!(v4d(8, 15, 7, -4), u);
    }

    #[test]
    fn test_sub_assign() {
        let mut u = v4d(2, 1, 5, 4);
        u -= v4d(3, 7, 1, -4);
        assert_eq!(v4d(-1, -6, 4, 8), u);
        u -= &v4d(3, 7, 1, -4);
        assert_eq!(v4d(-4, -13, 3, 12), u);
    }

    #[test]
    fn test_min() {
        let u = v4d(2, 1, 5, -4);
        let v = v4d(3, 7, 1, 4);
        assert_eq!(v4d(2, 1, 1, -4), u.min(v));
    }

    #[test]
    fn test_max() {
        let u = v4d(2, 1, 5, -4);
        let v = v4d(3, 7, 1, 4);
        assert_eq!(v4d(3, 7, 5, 4), u.max(v));
    }

    #[test]
    fn test_is_zero_true() {
        let v = v4d(0, 0, 0, 0);
        assert!(v.is_zero());
    }
    #[test]
    fn test_is_zero_false_x() {
        let v = v4d(1, 0, 0, 0);
        assert!(!v.is_zero());
    }
    #[test]
    fn test_is_zero_false_y() {
        let v = v4d(0, 1, 0, 0);
        assert!(!v.is_zero());
    }
    #[test]
    fn test_is_zero_false_z() {
        let v = v4d(0, 0, 1, 0);
        assert!(!v.is_zero());
    }
    #[test]
    fn test_is_zero_false_w() {
        let v = v4d(0, 0, 0, 1);
        assert!(!v.is_zero());
    }
}
