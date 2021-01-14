use core::cmp::Ordering;
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Point<S: Integer, V: Vector<S>> {
    s: PhantomData<S>,
    v: V,
}

pub type Point2d<S = i64> = Point<S, Vec2d<S>>;
pub type Point3d<S = i64> = Point<S, Vec3d<S>>;
pub type Point4d<S = i64> = Point<S, Vec4d<S>>;

impl<S, V> Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    pub fn with<F>(f: F) -> Point<S, V>
    where
        F: Fn(usize) -> S,
    {
        Point::from(V::with(f))
    }

    pub fn min(&self, other: Self) -> Self {
        Point::from(self.v.min(other.v))
    }
    pub fn max(&self, other: Self) -> Self {
        Point::from(self.v.max(other.v))
    }

    // The distance to another point w.r.t. the L1 norm.
    pub fn distance_l1(&self, other: Self) -> S {
        (self - other).norm_l1()
    }

    // The distance to another point w.r.t. the L∞ norm.
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
}

impl<S: Integer> Point2d<S> {
    /// Creates a new point with the given coordinates.
    pub fn new(x: S, y: S) -> Self {
        Point::from(v2d(x, y))
    }
    pub fn x(&self) -> S {
        self.v.x()
    }
    pub fn y(&self) -> S {
        self.v.y()
    }
}

impl<S: Integer> Point3d<S> {
    /// Creates a new point with the given coordinates.
    pub fn new(x: S, y: S, z: S) -> Self {
        Point::from(v3d(x, y, z))
    }
    pub fn x(&self) -> S {
        self.v.x()
    }
    pub fn y(&self) -> S {
        self.v.y()
    }
    pub fn z(&self) -> S {
        self.v.z()
    }
}

impl<S: Integer> Point4d<S> {
    /// Creates a new point with the given coordinates.
    pub fn new(x: S, y: S, z: S, w: S) -> Self {
        Point::from(v4d(x, y, z, w))
    }
    pub fn x(&self) -> S {
        self.v.x()
    }
    pub fn y(&self) -> S {
        self.v.y()
    }
    pub fn z(&self) -> S {
        self.v.z()
    }
    pub fn w(&self) -> S {
        self.v.w()
    }
}

/// Creates a point.
///
/// This is a utility function for concisely representing points.
pub fn p2d<S: Integer, T>(x: T, y: T) -> Point2d<S>
where
    S: From<T>,
{
    Point2d::new(S::from(x), S::from(y))
}
pub fn p3d<S: Integer, T>(x: T, y: T, z: T) -> Point3d<S>
where
    S: From<T>,
{
    Point3d::new(S::from(x), S::from(y), S::from(z))
}
pub fn p4d<S: Integer, T>(x: T, y: T, z: T, w: T) -> Point4d<S>
where
    S: From<T>,
{
    Point4d::new(S::from(x), S::from(y), S::from(z), S::from(w))
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

impl<S, V> PartialOrd for Point<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    fn partial_cmp(&self, other: &Point<S, V>) -> Option<Ordering> {
        self.v.partial_cmp(&other.v)
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
    use crate::Vec2d;
    use crate::Vec3d;
    use crate::Vec4d;

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
    fn test_index_2d() {
        let v = Point2d::new(1, 2);
        assert_eq!(1, v[0]);
        assert_eq!(2, v[1]);
    }
    #[test]
    fn test_index_3d() {
        let v = Point3d::new(1, 2, 3);
        assert_eq!(1, v[0]);
        assert_eq!(2, v[1]);
        assert_eq!(3, v[2]);
    }
    #[test]
    fn test_index_4d() {
        let v = Point4d::new(1, 2, 3, 4);
        assert_eq!(1, v[0]);
        assert_eq!(2, v[1]);
        assert_eq!(3, v[2]);
        assert_eq!(4, v[3]);
    }

    #[test]
    fn test_partial_ord_none() {
        let u = Point2d::new(2, 1);
        let v = Point2d::new(3, -7);
        assert_eq!(None, u.partial_cmp(&v));
    }
    #[test]
    fn test_partial_ord_less() {
        let u = Point2d::new(2, 1);
        let v = Point2d::new(3, 7);
        assert_eq!(Some(Ordering::Less), u.partial_cmp(&v));
    }
    #[test]
    fn test_partial_ord_equal() {
        let v = Point2d::new(3, 7);
        assert_eq!(Some(Ordering::Equal), v.partial_cmp(&v));
    }
    #[test]
    fn test_partial_ord_greater() {
        let u = Point2d::new(2, 1);
        let v = Point2d::new(3, 7);
        assert_eq!(Some(Ordering::Greater), v.partial_cmp(&u));
    }

    #[test]
    fn test_add_pv_2d() {
        let p: Point2d<i64> = Point2d::new(2, 1);
        let v = Vec2d::new(3, 7);
        assert_eq!(p2d(5, 8), p + v);
        assert_eq!(p2d(5, 8), p + &v);
        assert_eq!(p2d(5, 8), &p + v);
        assert_eq!(p2d(5, 8), &p + &v);
    }

    #[test]
    fn test_add_pv_3d() {
        let p: Point3d<i64> = Point3d::new(2, 1, 6);
        let v = Vec3d::new(3, 7, 1);
        assert_eq!(p3d(5, 8, 7), p + v);
        assert_eq!(p3d(5, 8, 7), p + &v);
        assert_eq!(p3d(5, 8, 7), &p + v);
        assert_eq!(p3d(5, 8, 7), &p + &v);
    }

    #[test]
    fn test_add_pv_4d() {
        let p: Point4d<i64> = Point4d::new(2, 1, 6, -2);
        let v = Vec4d::new(3, 7, 1, 5);
        assert_eq!(p4d(5, 8, 7, 3), p + v);
        assert_eq!(p4d(5, 8, 7, 3), p + &v);
        assert_eq!(p4d(5, 8, 7, 3), &p + v);
        assert_eq!(p4d(5, 8, 7, 3), &p + &v);
    }

    #[test]
    fn test_sub_pv() {
        let p = Point2d::new(2, 1);
        let v = Vec2d::new(3, 7);
        assert_eq!(p2d(-1, -6), p - v);
        assert_eq!(p2d(-1, -6), p - &v);
        assert_eq!(p2d(-1, -6), &p - v);
        assert_eq!(p2d(-1, -6), &p - &v);
    }

    #[test]
    fn test_sub_pp() {
        let p = Point2d::new(2, 1);
        let q = Point2d::new(3, 7);
        assert_eq!(v2d(-1, -6), p - q);
        assert_eq!(v2d(-1, -6), p - &q);
        assert_eq!(v2d(-1, -6), &p - q);
        assert_eq!(v2d(-1, -6), &p - &q);
    }

    #[test]
    fn test_add_assign() {
        let mut u = Point2d::new(2, 1);
        u += Vec2d::new(3, 7);
        assert_eq!(p2d(5, 8), u);
        u += &Vec2d::new(3, 7);
        assert_eq!(p2d(8, 15), u);
    }

    #[test]
    fn test_sub_assign() {
        let mut u = Point2d::new(2, 1);
        u -= Vec2d::new(3, 7);
        assert_eq!(p2d(-1, -6), u);
        u -= &Vec2d::new(3, 7);
        assert_eq!(p2d(-4, -13), u);
    }

    #[test]
    fn test_min_2d() {
        let u = Point2d::new(2, 7);
        let v = Point2d::new(3, 1);
        assert_eq!(p2d(2, 1), u.min(v));
    }

    #[test]
    fn test_min_3d() {
        let u = Point3d::new(2, 1, 5);
        let v = Point3d::new(3, 7, 1);
        assert_eq!(p3d(2, 1, 1), u.min(v));
    }

    #[test]
    fn test_min_4d() {
        let u = Point4d::new(2, 1, 5, -4);
        let v = Point4d::new(3, 7, 1, 4);
        assert_eq!(p4d(2, 1, 1, -4), u.min(v));
    }

    #[test]
    fn test_max_2d() {
        let u = Point2d::new(2, 7);
        let v = Point2d::new(3, 1);
        assert_eq!(p2d(3, 7), u.max(v));
    }

    #[test]
    fn test_max_3d() {
        let u = Point3d::new(2, 7, 5);
        let v = Point3d::new(3, 1, 1);
        assert_eq!(p3d(3, 7, 5), u.max(v));
    }

    #[test]
    fn test_max_4d() {
        let u = Point4d::new(2, 7, 5, -4);
        let v = Point4d::new(3, 1, 1, 4);
        assert_eq!(p4d(3, 7, 5, 4), u.max(v));
    }
}
