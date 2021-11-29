//! Points in the affine space.
//!
//! Points represent absolute positions in the space.
//! So in conmtrast to vectors it doesn't make sense to add two points,
//! or to multiply a point by a scalar value.
//!
//! A vector can be added to a point, representing a move in the space.
//! Or conversely, two points can be subtracted to get back the vector
//! representing the move between them.
//!
//! This module implements 2d, 3d and 4d points.

use core::cmp::Ordering;
use core::fmt;
use core::marker::PhantomData;

use core::ops::Add;
use core::ops::AddAssign;
use core::ops::Index;
use core::ops::Sub;
use core::ops::SubAssign;

use crate::v2d;
use crate::v3d;
use crate::v4d;
use crate::Integer;
use crate::Vec2d;
use crate::Vec3d;
use crate::Vec4d;
use crate::Vector;
use crate::VectorOps;

/// A point in a discrete space.
#[derive(Clone, Copy, Default, Eq, PartialEq, Hash)]
pub struct Point<S: Integer, V: Vector<S>> {
    s: PhantomData<S>,
    v: V,
}

/// A 2d point.
pub type Point2d<S = i64> = Point<S, Vec2d<S>>;
/// A 3d point.
pub type Point3d<S = i64> = Point<S, Vec3d<S>>;
/// A 4d point.
pub type Point4d<S = i64> = Point<S, Vec4d<S>>;

impl<S, V> Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    /// The dimension of the points in this type.
    pub const DIM: usize = V::DIM;

    /// Create a point from a function.
    ///
    /// The function must return a scalar value for each possible coordinate index.
    ///
    /// # Example
    /// ```
    /// # use std::convert::TryFrom;
    /// # use lowdim::p4d;
    /// # use lowdim::Point4d;
    /// assert_eq!(p4d(0, 1, 2, 3), Point4d::with(|i| i64::try_from(i).unwrap()));
    /// ```
    pub fn with<F>(f: F) -> Point<S, V>
    where
        F: Fn(usize) -> S,
    {
        Point::from(V::with(f))
    }

    /// Returns a vector of the point coordinates.
    pub fn to_vec(&self) -> V {
        self.v
    }

    /// Returns a slice containing the coordinates of the point.
    pub fn as_slice(&self) -> &[S] {
        self.v.as_slice()
    }

    /// Returns a mutable slice containing the coordinates of the point.
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        self.v.as_mut_slice()
    }

    /// Returns the point constructed by taking the minimum with another point at each coordinate.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p4d;
    /// let p0 = p4d(1, -2, 7, -4);
    /// let p1 = p4d(2, -3, 7, 4);
    /// assert_eq!(p4d(1, -3, 7, -4), p0.min(p1));
    /// ```
    pub fn min(&self, other: Self) -> Self {
        Point::from(self.v.min(other.v))
    }
    /// Returns the point constructed by taking the maximum with another point at each coordinate.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p4d;
    /// let p0 = p4d(1, -2, 7, -4);
    /// let p1 = p4d(2, -3, 7, 4);
    /// assert_eq!(p4d(2, -2, 7, 4), p0.max(p1));
    /// ```
    pub fn max(&self, other: Self) -> Self {
        Point::from(self.v.max(other.v))
    }

    /// Returns the distance to another point w.r.t. the L1 norm.
    ///
    /// This is the number of orthogonal steps needed
    /// to move from one point to the other.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p4d;
    /// let p0 = p4d(1, -2, 7, -4);
    /// let p1 = p4d(2, -3, 7, 4);
    /// assert_eq!(10, p0.distance_l1(p1));
    /// ```
    pub fn distance_l1(&self, other: Self) -> S {
        (self - other).norm_l1()
    }

    /// Returns the distance to another point w.r.t. the L∞ norm.
    ///
    /// This is the number of orthogonal or diagonal steps needed
    /// to move from one point to the other.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p4d;
    /// let p0 = p4d(1, -2, 7, -4);
    /// let p1 = p4d(2, -3, 7, 4);
    /// assert_eq!(8, p0.distance_l_infty(p1));
    /// ```
    pub fn distance_l_infty(&self, other: Self) -> S {
        (self - other).norm_l_infty()
    }

    /// Creates a vector containing the orthogonal neighbors of a point.
    pub fn neighbors_l1<'a>(&'a self) -> Vec<Self>
    where
        &'a V: VectorOps<S, V, V>,
    {
        V::unit_vecs_l1().into_iter().map(|v| self + v).collect()
    }

    /// Creates a vector containing the orthogonal and diagonal neighbors of a point.
    pub fn neighbors_l_infty<'a>(&'a self) -> Vec<Self>
    where
        &'a V: VectorOps<S, V, V>,
    {
        V::unit_vecs_l_infty()
            .into_iter()
            .map(|v| self + v)
            .collect()
    }

    /// Returns the componentwise partial ordering for this and another point.
    pub fn componentwise_cmp(&self, other: &Self) -> Option<Ordering> {
        self.v.componentwise_cmp(&other.v)
    }

    /// Returns the lexicographic total ordering for this and another point.
    ///
    /// That is, the first different coordinate decides the ordering.
    /// This is useful as an arbitrary total ordering for sorting,
    /// but is not intended to be otherwise meaningful.
    pub fn lex_cmp(&self, other: &Self) -> Ordering {
        self.v.lex_cmp(&other.v)
    }
}

impl<S: Integer> Point2d<S> {
    /// Creates a new point with the given coordinates.
    pub fn new(x: S, y: S) -> Self {
        Point::from(v2d(x, y))
    }
    /// Returns the x coordinate of the point.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::p2d;
    /// let p = p2d(2, 3);
    /// assert_eq!(2, p.x());
    /// ```
    pub fn x(&self) -> S {
        self.v.x()
    }
    /// Returns the y coordinate of the point.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::p2d;
    /// let p = p2d(2, 3);
    /// assert_eq!(3, p.y());
    /// ```
    pub fn y(&self) -> S {
        self.v.y()
    }
}

impl<S: Integer> Point3d<S> {
    /// Creates a new point with the given coordinates.
    pub fn new(x: S, y: S, z: S) -> Self {
        Point::from(v3d(x, y, z))
    }
    /// Returns the x coordinate of the point.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::p3d;
    /// let p = p3d(2, 3, -1);
    /// assert_eq!(2, p.x());
    /// ```
    pub fn x(&self) -> S {
        self.v.x()
    }
    /// Returns the y coordinate of the point.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::p3d;
    /// let p = p3d(2, 3, -1);
    /// assert_eq!(3, p.y());
    /// ```
    pub fn y(&self) -> S {
        self.v.y()
    }
    /// Returns the z coordinate of the point.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::p3d;
    /// let p = p3d(2, 3, -1);
    /// assert_eq!(-1, p.z());
    /// ```
    pub fn z(&self) -> S {
        self.v.z()
    }
}

impl<S: Integer> Point4d<S> {
    /// Creates a new point with the given coordinates.
    pub fn new(x: S, y: S, z: S, w: S) -> Self {
        Point::from(v4d(x, y, z, w))
    }

    /// Returns the x coordinate of the point.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::p4d;
    /// let p = p4d(2, 3, -1, 4);
    /// assert_eq!(2, p.x());
    /// ```
    pub fn x(&self) -> S {
        self.v.x()
    }
    /// Returns the y coordinate of the point.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::p4d;
    /// let p = p4d(2, 3, -1, 4);
    /// assert_eq!(3, p.y());
    /// ```
    pub fn y(&self) -> S {
        self.v.y()
    }
    /// Returns the z coordinate of the point.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::p4d;
    /// let p = p4d(2, 3, -1, 4);
    /// assert_eq!(-1, p.z());
    /// ```
    pub fn z(&self) -> S {
        self.v.z()
    }
    /// Returns the w coordinate of the point.
    ///
    /// # Examples
    /// ```
    /// # use lowdim::p4d;
    /// let p = p4d(2, 3, -1, 4);
    /// assert_eq!(4, p.w());
    /// ```
    pub fn w(&self) -> S {
        self.v.w()
    }
}

/// Creates a 2d point.
///
/// This is a utility function for concisely representing points.
pub fn p2d<S: Integer>(x: S, y: S) -> Point2d<S> {
    Point2d::new(x, y)
}
/// Creates a 3d point.
///
/// This is a utility function for concisely representing points.
pub fn p3d<S: Integer>(x: S, y: S, z: S) -> Point3d<S> {
    Point3d::new(x, y, z)
}
/// Creates a 4d point.
///
/// This is a utility function for concisely representing points.
pub fn p4d<S: Integer>(x: S, y: S, z: S, w: S) -> Point4d<S> {
    Point4d::new(x, y, z, w)
}

impl<S, V> fmt::Debug for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P(")?;
        let mut sep = "";
        for i in 0..Self::DIM {
            write!(f, "{}{}", sep, self[i])?;
            sep = ", ";
        }
        write!(f, ")")
    }
}

impl<S, V> From<V> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn from(v: V) -> Self {
        Point { s: PhantomData, v }
    }
}

impl<S, V> Index<usize> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = S;
    fn index(&self, i: usize) -> &S {
        self.v.index(i)
    }
}

impl<S, V> Add<V> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = Point<S, V>;

    fn add(self, other: V) -> Point<S, V> {
        Point::from(self.v + other)
    }
}
impl<'a, S, V> Add<&'a V> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = Point<S, V>;

    fn add(self, other: &'a V) -> Point<S, V> {
        Point::from(self.v + other)
    }
}
impl<'a, S, V> Add<V> for &'a Point<S, V>
where
    S: Integer,
    V: Vector<S>,
    &'a V: VectorOps<S, V, V>,
{
    type Output = Point<S, V>;

    fn add(self, other: V) -> Point<S, V> {
        Point::from(self.v + other)
    }
}
impl<'a, S, V> Add<&'a V> for &'a Point<S, V>
where
    S: Integer,
    V: Vector<S>,
    &'a V: VectorOps<S, &'a V, V>,
{
    type Output = Point<S, V>;

    fn add(self, other: &'a V) -> Point<S, V> {
        Point::from(self.v + other)
    }
}

impl<S, V> Sub<V> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = Point<S, V>;

    fn sub(self, other: V) -> Point<S, V> {
        Point::from(self.v - other)
    }
}
impl<'a, S, V> Sub<&'a V> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = Point<S, V>;

    fn sub(self, other: &'a V) -> Point<S, V> {
        Point::from(self.v - *other)
    }
}
impl<'a, S, V> Sub<V> for &'a Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = Point<S, V>;

    fn sub(self, other: V) -> Point<S, V> {
        Point::from(self.v - other)
    }
}
impl<'a, S, V> Sub<&'a V> for &'a Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = Point<S, V>;

    fn sub(self, other: &'a V) -> Point<S, V> {
        Point::from(self.v - *other)
    }
}

impl<S, V> Sub<Point<S, V>> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = V;

    fn sub(self, other: Point<S, V>) -> V {
        self.v - other.v
    }
}
impl<'a, S, V> Sub<&'a Point<S, V>> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = V;

    fn sub(self, other: &'a Point<S, V>) -> V {
        self.v - other.v
    }
}
impl<'a, S, V> Sub<Point<S, V>> for &'a Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = V;

    fn sub(self, other: Point<S, V>) -> V {
        self.v - other.v
    }
}
impl<'a, S, V> Sub<&'a Point<S, V>> for &'a Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    type Output = V;

    fn sub(self, other: &'a Point<S, V>) -> V {
        self.v - other.v
    }
}

impl<S, V> AddAssign<V> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn add_assign(&mut self, other: V) {
        *self = Point::from(self.v + other);
    }
}
impl<'a, S, V> AddAssign<&'a V> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn add_assign(&mut self, other: &'a V) {
        *self = Point::from(self.v + *other);
    }
}

impl<S, V> SubAssign<V> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn sub_assign(&mut self, other: V) {
        *self = Point::from(self.v - other);
    }
}
impl<'a, S, V> SubAssign<&'a V> for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn sub_assign(&mut self, other: &'a V) {
        *self = Point::from(self.v - *other);
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;
    use core::convert::TryFrom;

    use crate::p2d;
    use crate::p3d;
    use crate::p4d;
    use crate::v2d;
    use crate::v3d;
    use crate::v4d;
    use crate::Point2d;
    use crate::Point3d;
    use crate::Point4d;

    #[test]
    fn test_from_2d() {
        let p: Point2d<i64> = Point2d::from(v2d(1, 2));
        assert_eq!(p2d(1, 2), p);
    }

    #[test]
    fn test_from_3d() {
        let p: Point3d<i64> = Point3d::from(v3d(1, 2, 3));
        assert_eq!(p3d(1, 2, 3), p);
    }

    #[test]
    fn test_from_4d() {
        let p: Point4d<i64> = Point4d::from(v4d(1, 2, 3, 4));
        assert_eq!(p4d(1, 2, 3, 4), p);
    }

    #[test]
    fn test_from_i32() {
        let p: Point2d<i32> = Point2d::from(v2d(1, 2));
        assert_eq!(p2d(1, 2), p);
    }

    #[test]
    fn test_xy_2d() {
        let p: Point2d<i64> = Point2d::from(v2d(1, 2));
        assert_eq!(1, p.x());
        assert_eq!(2, p.y());
    }

    #[test]
    fn test_xyz_3d() {
        let p: Point3d<i64> = Point3d::from(v3d(1, 2, 3));
        assert_eq!(1, p.x());
        assert_eq!(2, p.y());
        assert_eq!(3, p.z());
    }

    #[test]
    fn test_xyzw_4d() {
        let p: Point4d<i64> = Point4d::from(v4d(1, 2, 3, 4));
        assert_eq!(1, p.x());
        assert_eq!(2, p.y());
        assert_eq!(3, p.z());
        assert_eq!(4, p.w());
    }

    #[test]
    fn test_with() {
        assert_eq!(p2d(2, 3), Point2d::with(|i| i64::try_from(i + 2).unwrap()));
    }

    #[test]
    fn test_as_slice() {
        assert_eq!(&[2, 3], p2d(2, 3).as_slice());
    }
    #[test]
    fn test_as_mut_slice() {
        let mut v = p2d(2, 3);
        let s = v.as_mut_slice();
        s[1] += 1;
        assert_eq!(&[2, 4], s);
    }

    #[test]
    fn test_debug() {
        assert_eq!("P(2, -3)", format!("{:?}", p2d(2, -3)));
    }

    #[test]
    fn test_index_2d() {
        let v = p2d(1, 2);
        assert_eq!(1, v[0]);
        assert_eq!(2, v[1]);
    }
    #[test]
    fn test_index_3d() {
        let v = p3d(1, 2, 3);
        assert_eq!(1, v[0]);
        assert_eq!(2, v[1]);
        assert_eq!(3, v[2]);
    }
    #[test]
    fn test_index_4d() {
        let v = p4d(1, 2, 3, 4);
        assert_eq!(1, v[0]);
        assert_eq!(2, v[1]);
        assert_eq!(3, v[2]);
        assert_eq!(4, v[3]);
    }

    #[test]
    fn test_componentwise_cmp_none() {
        let u = p2d(2, 1);
        let v = p2d(3, -7);
        assert_eq!(None, u.componentwise_cmp(&v));
    }
    #[test]
    fn test_componentwise_cmp_less() {
        let u = p2d(2, 1);
        let v = p2d(3, 7);
        assert_eq!(Some(Ordering::Less), u.componentwise_cmp(&v));
    }
    #[test]
    fn test_componentwise_cmp_equal() {
        let v = p2d(3, 7);
        assert_eq!(Some(Ordering::Equal), v.componentwise_cmp(&v));
    }
    #[test]
    fn test_componentwise_cmp_greater() {
        let u = p2d(2, 1);
        let v = p2d(3, 7);
        assert_eq!(Some(Ordering::Greater), v.componentwise_cmp(&u));
    }

    #[test]
    fn test_add_pv_2d() {
        let p: Point2d<i64> = p2d(2, 1);
        let v = v2d(3, 7);
        assert_eq!(p2d(5, 8), p + v);
        assert_eq!(p2d(5, 8), p + &v);
        assert_eq!(p2d(5, 8), &p + v);
        assert_eq!(p2d(5, 8), &p + &v);
    }

    #[test]
    fn test_add_pv_3d() {
        let p: Point3d<i64> = p3d(2, 1, 6);
        let v = v3d(3, 7, 1);
        assert_eq!(p3d(5, 8, 7), p + v);
        assert_eq!(p3d(5, 8, 7), p + &v);
        assert_eq!(p3d(5, 8, 7), &p + v);
        assert_eq!(p3d(5, 8, 7), &p + &v);
    }

    #[test]
    fn test_add_pv_4d() {
        let p: Point4d<i64> = p4d(2, 1, 6, -2);
        let v = v4d(3, 7, 1, 5);
        assert_eq!(p4d(5, 8, 7, 3), p + v);
        assert_eq!(p4d(5, 8, 7, 3), p + &v);
        assert_eq!(p4d(5, 8, 7, 3), &p + v);
        assert_eq!(p4d(5, 8, 7, 3), &p + &v);
    }

    #[test]
    fn test_sub_pv() {
        let p = p2d(2, 1);
        let v = v2d(3, 7);
        assert_eq!(p2d(-1, -6), p - v);
        assert_eq!(p2d(-1, -6), p - &v);
        assert_eq!(p2d(-1, -6), &p - v);
        assert_eq!(p2d(-1, -6), &p - &v);
    }

    #[test]
    fn test_sub_pp() {
        let p = p2d(2, 1);
        let q = p2d(3, 7);
        assert_eq!(v2d(-1, -6), p - q);
        assert_eq!(v2d(-1, -6), p - &q);
        assert_eq!(v2d(-1, -6), &p - q);
        assert_eq!(v2d(-1, -6), &p - &q);
    }

    #[test]
    fn test_add_assign() {
        let mut u = p2d(2, 1);
        u += v2d(3, 7);
        assert_eq!(p2d(5, 8), u);
        u += &v2d(3, 7);
        assert_eq!(p2d(8, 15), u);
    }

    #[test]
    fn test_sub_assign() {
        let mut u = p2d(2, 1);
        u -= v2d(3, 7);
        assert_eq!(p2d(-1, -6), u);
        u -= &v2d(3, 7);
        assert_eq!(p2d(-4, -13), u);
    }

    #[test]
    fn test_min_2d() {
        let u = p2d(2, 7);
        let v = p2d(3, 1);
        assert_eq!(p2d(2, 1), u.min(v));
    }

    #[test]
    fn test_min_3d() {
        let u = p3d(2, 1, 5);
        let v = p3d(3, 7, 1);
        assert_eq!(p3d(2, 1, 1), u.min(v));
    }

    #[test]
    fn test_min_4d() {
        let u = p4d(2, 1, 5, -4);
        let v = p4d(3, 7, 1, 4);
        assert_eq!(p4d(2, 1, 1, -4), u.min(v));
    }

    #[test]
    fn test_max_2d() {
        let u = p2d(2, 7);
        let v = p2d(3, 1);
        assert_eq!(p2d(3, 7), u.max(v));
    }

    #[test]
    fn test_max_3d() {
        let u = p3d(2, 7, 5);
        let v = p3d(3, 1, 1);
        assert_eq!(p3d(3, 7, 5), u.max(v));
    }

    #[test]
    fn test_max_4d() {
        let u = p4d(2, 7, 5, -4);
        let v = p4d(3, 1, 1, 4);
        assert_eq!(p4d(3, 7, 5, 4), u.max(v));
    }

    #[test]
    fn test_neighbors_l1() {
        let p = p2d(2, -3);
        let mut ns: Vec<Point2d<i64>> = p.neighbors_l1();
        ns.sort_by(Point2d::lex_cmp);
        assert_eq!(vec![p2d(1, -3), p2d(2, -4), p2d(2, -2), p2d(3, -3)], ns);
    }

    #[test]
    fn test_unit_vecs_l_infty() {
        let p = p2d(2, -3);
        let mut ns: Vec<Point2d<i64>> = p.neighbors_l_infty();
        ns.sort_by(Point2d::lex_cmp);
        assert_eq!(
            vec![
                p2d(1, -4),
                p2d(1, -3),
                p2d(1, -2),
                p2d(2, -4),
                p2d(2, -2),
                p2d(3, -4),
                p2d(3, -3),
                p2d(3, -2)
            ],
            ns
        );
    }
}
