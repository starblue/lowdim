use core::marker::PhantomData;

use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;

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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
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
    fn from(v: V) -> Self {
        Point { s: PhantomData, v }
    }

    pub fn min(&self, other: Self) -> Self {
        Point::from(self.v.min(other.v))
    }
    pub fn max(&self, other: Self) -> Self {
        Point::from(self.v.max(other.v))
    }

    /// Creates a vector containing the orthogonal neighbours of a point.
    pub fn neighbours_l1<'a>(&'a self) -> Vec<Self>
    where
        &'a V: VectorOps<S, V>,
    {
        V::unit_vecs_l1().into_iter().map(|v| self + v).collect()
    }

    /// Creates a vector containing the orthogonal and diagonal neighbours of a point.
    pub fn neighbours_l_infty<'a>(&'a self) -> Vec<Self>
    where
        &'a V: VectorOps<S, V>,
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
    &'a V: VectorOps<S, V>,
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
    &'a V: VectorOps<S, &'a V>,
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
