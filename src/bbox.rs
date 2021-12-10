//! n-dimensional bounding boxes.

use core::fmt;
use core::ops::Bound;
use core::ops::Range;
use core::ops::RangeBounds;
use core::ops::RangeInclusive;

#[cfg(feature = "random")]
use rand::distributions::uniform::SampleUniform;
#[cfg(feature = "random")]
use rand::Rng;

use crate::layout::Points;
use crate::p2d;
use crate::p3d;
use crate::p4d;
use crate::Integer;
use crate::Layout;
use crate::Layout2d;
use crate::Layout3d;
use crate::Layout4d;
use crate::Point;
use crate::Point2d;
use crate::Point3d;
use crate::Point4d;
use crate::Vec2d;
use crate::Vec3d;
use crate::Vec4d;
use crate::Vector;

/// A bounding box.
///
/// An axis-aligned non-empty volume of the same dimension as the space.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BBox<S: Integer, V: Vector<S>> {
    /// The corner of the box with the lowest coordinates.
    min: Point<S, V>,
    /// The corner of the box with the highest coordinates.
    max: Point<S, V>,
}

/// A 2d bounding box.
pub type BBox2d<S = i64> = BBox<S, Vec2d<S>>;
/// A 3d bounding box.
pub type BBox3d<S = i64> = BBox<S, Vec3d<S>>;
/// A 4d bounding box.
pub type BBox4d<S = i64> = BBox<S, Vec4d<S>>;

impl<S, V> BBox<S, V>
where
    S: Integer,
    V: Vector<S>,
{
    /// Constructs a bounding box from origin and size.
    ///
    /// All coordinates of `size` must be positive to make the box non-empty.
    pub fn new(origin: Point<S, V>, size: V) -> BBox<S, V> {
        let min = origin;
        let max = Point::with(|i| origin[i] + size[i] - S::one());
        assert!(size.as_slice().iter().all(|&c| c > S::zero()));
        BBox { min, max }
    }
    /// Constructs a bounding box which contains a single point.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let p = p2d(2, 3);
    /// assert_eq!(bb2d(2..3, 3..4), BBox2d::from_point(p));
    /// ```
    pub fn from_point(p: Point<S, V>) -> BBox<S, V> {
        BBox { min: p, max: p }
    }
    /// Constructs the smallest bounding box containing two points.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let p0 = p2d(2, 3);
    /// let p1 = p2d(-1, 4);
    /// assert_eq!(bb2d(-1..3, 3..5), BBox2d::from_points(p0, p1));
    /// ```
    pub fn from_points(p0: Point<S, V>, p1: Point<S, V>) -> BBox<S, V> {
        let min = p0.min(p1);
        let max = p0.max(p1);
        BBox { min, max }
    }
    /// The minimal point in the bounding box.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let b = bb2d(-1..3, 3..5);
    /// assert_eq!(p2d(-1, 3), b.min());
    /// ```
    pub fn min(&self) -> Point<S, V> {
        self.min
    }
    /// The maximal point in the bounding box.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let b = bb2d(-1..3, 3..5);
    /// assert_eq!(p2d(2, 4), b.max());
    /// ```
    pub fn max(&self) -> Point<S, V> {
        self.max
    }

    /// Returns a vector of the lengths for the coordinates.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let b = bb2d(-1..3, 3..5);
    /// assert_eq!(p2d(2, 4), b.max());
    /// ```
    pub fn lengths(&self) -> V {
        self.max - self.min + V::ones()
    }

    /// Returns the volume of the bounding box.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let b = bb2d(-1..3, 3..5);
    /// assert_eq!(8, b.volume());
    /// ```
    pub fn volume(&self) -> usize
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        self.lengths()
            .as_slice()
            .iter()
            .map(|&len| len.to_usize())
            .product()
    }

    /// The center point in the bounding box.
    ///
    /// This is only the true center of the bounding box
    /// if the bounding box has odd dimensions.
    /// Otherwise the coordinates of the center are rounded
    /// according to the rules for integer division, i.e. towards zero.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let b = bb2d(-1..4, 3..6);
    /// assert_eq!(p2d(1, 4), b.center());
    /// ```
    pub fn center(&self) -> Point<S, V> {
        Point::<S, V>::from((self.min.to_vec() + self.max.to_vec()) / S::two())
    }

    /// The least upper bound of two bounding boxes w.r.t. inclusion.
    ///
    /// That is, the smallest bounding box encompassing the points in the two boxes.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let bb0 = bb2d(-2..3, -1..2);
    /// let bb1 = bb2d(-5..4, 2..5);
    /// assert_eq!(bb2d(-5..4, -1..5), bb0.lub(&bb1));
    /// ```
    pub fn lub(&self, other: &BBox<S, V>) -> BBox<S, V> {
        let min = self.min().min(other.min());
        let max = self.max().max(other.max());
        BBox { min, max }
    }

    /// Returns the smallest bounding box encompassing this box and a given point.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let bb = bb2d(-2..3, -1..2);
    /// let p = p2d(3, 4);
    /// assert_eq!(bb2d(-2..4, -1..5), bb.extend_to(p));
    /// ```
    pub fn extend_to(&self, p: Point<S, V>) -> BBox<S, V> {
        let min = self.min().min(p);
        let max = self.max().max(p);
        BBox { min, max }
    }

    /// Returns the closest point inside the bounding box for a given point.
    ///
    /// If the point is inside the bounding box it is returned unchanged.
    /// Otherwise the coordinates that fall outside the ranges of the box
    /// are clamped to the closest endpoints of the ranges.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let b = bb2d(-1..4, 3..6);
    /// assert_eq!(p2d(3, 4), b.clamp(p2d(10, 4)));
    /// ```
    pub fn clamp(&self, p: Point<S, V>) -> Point<S, V> {
        p.max(self.min).min(self.max)
    }

    /// Returns true if the point is inside the bounding box.
    ///
    /// # Example
    /// ```
    /// # use lowdim::p2d;
    /// # use lowdim::bb2d;
    /// # use lowdim::BBox2d;
    /// let b = bb2d(-1..4, 3..6);
    /// assert!(b.contains(&p2d(0, 4)));
    /// assert!(!b.contains(&p2d(10, 4)));
    /// ```
    pub fn contains(&self, p: &Point<S, V>) -> bool {
        self.clamp(*p) == *p
    }
}

impl<S> BBox2d<S>
where
    S: Integer,
{
    /// Constructs a bounding box from bounds.
    ///
    /// As always, lower bbox are inclusive, upper bbox exclusive.
    #[deprecated = "Use `bb2d` instead."]
    pub fn from_bounds(x_start: S, x_end: S, y_start: S, y_end: S) -> BBox2d<S> {
        assert!(x_start <= x_end && y_start <= y_end);
        let min = p2d(x_start, y_start);
        let max = p2d(x_end - S::one(), y_end - S::one());
        BBox { min, max }
    }

    /// Returns the lower bound in the x-coordinate (inclusive).
    pub fn x_start(&self) -> S {
        self.min.x()
    }
    /// Returns the upper bound in the x-coordinate (exclusive).
    pub fn x_end(&self) -> S {
        self.max.x() + S::one()
    }
    /// Returns the lower bound in the x-coordinate (inclusive).
    pub fn x_min(&self) -> S {
        self.min.x()
    }
    /// Returns the upper bound in the x-coordinate (inclusive).
    pub fn x_max(&self) -> S {
        self.max.x()
    }

    /// Returns the lower bound in the y-coordinate (inclusive).
    pub fn y_start(&self) -> S {
        self.min.y()
    }
    /// Returns the upper bound in the y-coordinate (exclusive).
    pub fn y_end(&self) -> S {
        self.max.y() + S::one()
    }
    /// Returns the lower bound in the y-coordinate (inclusive).
    pub fn y_min(&self) -> S {
        self.min.y()
    }
    /// Returns the upper bound in the y-coordinate (inclusive).
    pub fn y_max(&self) -> S {
        self.max.y()
    }

    /// Returns the width of the bounding box.
    pub fn x_len(&self) -> S {
        self.max.x() - self.min.x() + S::one()
    }

    /// Returns the height of the bounding box.
    pub fn y_len(&self) -> S {
        self.max.y() - self.min.y() + S::one()
    }

    /// Returns the area of the bounding box.
    pub fn area(&self) -> usize
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        self.volume()
    }

    /// The range of the x coordinate
    pub fn x_range(&self) -> Range<S> {
        self.x_start()..self.x_end()
    }
    /// The range of the y coordinate
    pub fn y_range(&self) -> Range<S> {
        self.y_start()..self.y_end()
    }

    /// Returns an iterator over the points in the bounding box.
    ///
    /// Points are returned by row.
    pub fn iter(&self) -> Points<S, Vec2d<S>, Layout2d<S>>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        Layout2d::new(*self).points()
    }

    /// Returns a random point inside the bounding box.
    ///
    /// Uses a uniform distribution.
    #[cfg(feature = "random")]
    pub fn random_point<R>(&self, rng: &mut R) -> Point2d<S>
    where
        R: Rng,
        S: SampleUniform,
    {
        let x = rng.gen_range(self.x_range());
        let y = rng.gen_range(self.y_range());
        p2d(x, y)
    }
}

fn to_range_inclusive<S, R>(r: R) -> RangeInclusive<S>
where
    S: Integer,
    R: RangeBounds<S>,
{
    let start = match r.start_bound() {
        Bound::Included(s) => *s,
        Bound::Excluded(s) => *s + S::one(),
        Bound::Unbounded => panic!("unbounded ranges are not allowed"),
    };
    let end = match r.end_bound() {
        Bound::Included(s) => *s,
        Bound::Excluded(s) => *s - S::one(),
        Bound::Unbounded => panic!("unbounded ranges are not allowed"),
    };
    start..=end
}

/// Creates a 2d bounding box from ranges of x and y coordinates.
pub fn bb2d<S, RX, RY>(rx: RX, ry: RY) -> BBox2d<S>
where
    S: Integer,
    RX: RangeBounds<S>,
    RY: RangeBounds<S>,
{
    let rx = to_range_inclusive(rx);
    let ry = to_range_inclusive(ry);
    let p_min = p2d(*rx.start(), *ry.start());
    let p_max = p2d(*rx.end(), *ry.end());
    BBox2d::from_points(p_min, p_max)
}
/// Creates a 3d bounding box from ranges of x, y and z coordinates.
pub fn bb3d<S, RX, RY, RZ>(rx: RX, ry: RY, rz: RZ) -> BBox3d<S>
where
    S: Integer,
    RX: RangeBounds<S>,
    RY: RangeBounds<S>,
    RZ: RangeBounds<S>,
{
    let rx = to_range_inclusive(rx);
    let ry = to_range_inclusive(ry);
    let rz = to_range_inclusive(rz);
    let p_min = p3d(*rx.start(), *ry.start(), *rz.start());
    let p_max = p3d(*rx.end(), *ry.end(), *rz.end());
    BBox3d::from_points(p_min, p_max)
}
/// Creates a 4d bounding box from ranges of x, y, z and w coordinates.
pub fn bb4d<S, RX, RY, RZ, RW>(rx: RX, ry: RY, rz: RZ, rw: RW) -> BBox4d<S>
where
    S: Integer,
    RX: RangeBounds<S>,
    RY: RangeBounds<S>,
    RZ: RangeBounds<S>,
    RW: RangeBounds<S>,
{
    let rx = to_range_inclusive(rx);
    let ry = to_range_inclusive(ry);
    let rz = to_range_inclusive(rz);
    let rw = to_range_inclusive(rw);
    let p_min = p4d(*rx.start(), *ry.start(), *rz.start(), *rw.start());
    let p_max = p4d(*rx.end(), *ry.end(), *rz.end(), *rw.end());
    BBox4d::from_points(p_min, p_max)
}

impl<S> BBox3d<S>
where
    S: Integer,
{
    /// Returns the lower bound in the x-coordinate (inclusive).
    pub fn x_start(&self) -> S {
        self.min.x()
    }
    /// Returns the upper bound in the x-coordinate (exclusive).
    pub fn x_end(&self) -> S {
        self.max.x() + S::one()
    }
    /// Returns the lower bound in the x-coordinate (inclusive).
    pub fn x_min(&self) -> S {
        self.min.x()
    }
    /// Returns the upper bound in the x-coordinate (inclusive).
    pub fn x_max(&self) -> S {
        self.max.x()
    }

    /// Returns the lower bound in the y-coordinate (inclusive).
    pub fn y_start(&self) -> S {
        self.min.y()
    }
    /// Returns the upper bound in the y-coordinate (exclusive).
    pub fn y_end(&self) -> S {
        self.max.y() + S::one()
    }
    /// Returns the lower bound in the y-coordinate (inclusive).
    pub fn y_min(&self) -> S {
        self.min.y()
    }
    /// Returns the upper bound in the y-coordinate (inclusive).
    pub fn y_max(&self) -> S {
        self.max.y()
    }

    /// Returns the lower bound in the z-coordinate (inclusive).
    pub fn z_start(&self) -> S {
        self.min.z()
    }
    /// Returns the upper bound in the z-coordinate (exclusive).
    pub fn z_end(&self) -> S {
        self.max.z() + S::one()
    }
    /// Returns the lower bound in the z-coordinate (inclusive).
    pub fn z_min(&self) -> S {
        self.min.z()
    }
    /// Returns the upper bound in the z-coordinate (inclusive).
    pub fn z_max(&self) -> S {
        self.max.z()
    }

    /// Returns the size of the x-dimension of the bounding box.
    pub fn x_len(&self) -> S {
        self.max.x() - self.min.x() + S::one()
    }
    /// Returns the size of the y-dimension of the bounding box.
    pub fn y_len(&self) -> S {
        self.max.y() - self.min.y() + S::one()
    }
    /// Returns the size of the z-dimension of the bounding box.
    pub fn z_len(&self) -> S {
        self.max.z() - self.min.z() + S::one()
    }

    /// Returns the range of the x coordinate.
    pub fn x_range(&self) -> Range<S> {
        self.x_start()..self.x_end()
    }
    /// Returns the range of the y coordinate.
    pub fn y_range(&self) -> Range<S> {
        self.y_start()..self.y_end()
    }
    /// Returns the range of the z coordinate.
    pub fn z_range(&self) -> Range<S> {
        self.z_start()..self.z_end()
    }

    /// Returns an iterator over the points in the bounding box.
    ///
    /// Points are returned by row.
    pub fn iter(&self) -> Points<S, Vec3d<S>, Layout3d<S>>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        Layout3d::new(*self).points()
    }

    /// Returns a random point inside the bounding box.
    ///
    /// Uses a uniform distribution.
    #[cfg(feature = "random")]
    pub fn random_point<R>(&self, rng: &mut R) -> Point3d<S>
    where
        R: Rng,
        S: SampleUniform,
    {
        let x = rng.gen_range(self.x_range());
        let y = rng.gen_range(self.y_range());
        let z = rng.gen_range(self.z_range());
        p3d(x, y, z)
    }
}

impl<S> BBox4d<S>
where
    S: Integer,
{
    /// Returns the lower bound in the x-coordinate (inclusive).
    pub fn x_start(&self) -> S {
        self.min.x()
    }
    /// Returns the upper bound in the x-coordinate (exclusive).
    pub fn x_end(&self) -> S {
        self.max.x() + S::one()
    }
    /// Returns the lower bound in the x-coordinate (inclusive).
    pub fn x_min(&self) -> S {
        self.min.x()
    }
    /// Returns the upper bound in the x-coordinate (inclusive).
    pub fn x_max(&self) -> S {
        self.max.x()
    }

    /// Returns the lower bound in the y-coordinate (inclusive).
    pub fn y_start(&self) -> S {
        self.min.y()
    }
    /// Returns the upper bound in the y-coordinate (exclusive).
    pub fn y_end(&self) -> S {
        self.max.y() + S::one()
    }
    /// Returns the lower bound in the y-coordinate (inclusive).
    pub fn y_min(&self) -> S {
        self.min.y()
    }
    /// Returns the upper bound in the y-coordinate (inclusive).
    pub fn y_max(&self) -> S {
        self.max.y()
    }

    /// Returns the lower bound in the z-coordinate (inclusive).
    pub fn z_start(&self) -> S {
        self.min.z()
    }
    /// Returns the upper bound in the z-coordinate (exclusive).
    pub fn z_end(&self) -> S {
        self.max.z() + S::one()
    }
    /// Returns the lower bound in the z-coordinate (inclusive).
    pub fn z_min(&self) -> S {
        self.min.z()
    }
    /// Returns the upper bound in the z-coordinate (inclusive).
    pub fn z_max(&self) -> S {
        self.max.z()
    }

    /// Returns the lower bound in the w-coordinate (inclusive).
    pub fn w_start(&self) -> S {
        self.min.w()
    }
    /// Returns the upper bound in the w-coordinate (exclusive).
    pub fn w_end(&self) -> S {
        self.max.w() + S::one()
    }
    /// Returns the lower bound in the w-coordinate (inclusive).
    pub fn w_min(&self) -> S {
        self.min.w()
    }
    /// Returns the upper bound in the w-coordinate (inclusive).
    pub fn w_max(&self) -> S {
        self.max.w()
    }

    /// Returns the size of the x-dimension of the bounding box.
    pub fn x_len(&self) -> S {
        self.max.x() - self.min.x() + S::one()
    }
    /// Returns the size of the y-dimension of the bounding box.
    pub fn y_len(&self) -> S {
        self.max.y() - self.min.y() + S::one()
    }
    /// Returns the size of the z-dimension of the bounding box.
    pub fn z_len(&self) -> S {
        self.max.z() - self.min.z() + S::one()
    }
    /// Returns the size of the w-dimension of the bounding box.
    pub fn w_len(&self) -> S {
        self.max.w() - self.min.w() + S::one()
    }

    /// Returns the range of the x coordinate.
    pub fn x_range(&self) -> Range<S> {
        self.x_start()..self.x_end()
    }
    /// Returns the range of the y coordinate.
    pub fn y_range(&self) -> Range<S> {
        self.y_start()..self.y_end()
    }
    /// Returns the range of the z coordinate.
    pub fn z_range(&self) -> Range<S> {
        self.z_start()..self.z_end()
    }
    /// Returns the range of the w coordinate.
    pub fn w_range(&self) -> Range<S> {
        self.w_start()..self.w_end()
    }

    /// Returns an iterator over the points in the bounding box.
    ///
    /// Points are returned by row.
    pub fn iter(&self) -> Points<S, Vec4d<S>, Layout4d<S>>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        Layout4d::new(*self).points()
    }

    /// Returns a random point inside the bounding box.
    ///
    /// Uses a uniform distribution.
    #[cfg(feature = "random")]
    pub fn random_point<R>(&self, rng: &mut R) -> Point4d<S>
    where
        R: Rng,
        S: SampleUniform,
    {
        let x = rng.gen_range(self.x_range());
        let y = rng.gen_range(self.y_range());
        let z = rng.gen_range(self.z_range());
        let w = rng.gen_range(self.w_range());
        p4d(x, y, z, w)
    }
}

impl<'a, S> IntoIterator for &'a BBox<S, Vec2d<S>>
where
    S: Integer,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    type Item = Point2d<S>;

    type IntoIter = Points<S, Vec2d<S>, Layout2d<S>>;

    fn into_iter(self) -> Self::IntoIter {
        Layout2d::new(*self).into_points()
    }
}
impl<'a, S> IntoIterator for &'a BBox<S, Vec3d<S>>
where
    S: Integer,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    type Item = Point3d<S>;

    type IntoIter = Points<S, Vec3d<S>, Layout3d<S>>;

    fn into_iter(self) -> Self::IntoIter {
        Layout3d::new(*self).into_points()
    }
}
impl<'a, S> IntoIterator for &'a BBox<S, Vec4d<S>>
where
    S: Integer,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    type Item = Point4d<S>;

    type IntoIter = Points<S, Vec4d<S>, Layout4d<S>>;

    fn into_iter(self) -> Self::IntoIter {
        Layout4d::new(*self).into_points()
    }
}

#[cfg(test)]
mod tests {
    use crate::bb2d;
    use crate::bb3d;
    use crate::bb4d;
    use crate::p2d;
    use crate::p3d;
    use crate::p4d;
    use crate::v2d;
    use crate::v3d;
    use crate::v4d;
    use crate::BBox2d;

    #[cfg(feature = "random")]
    use rand::rngs::StdRng;
    #[cfg(feature = "random")]
    use rand::SeedableRng;

    #[test]
    fn test_new() {
        let bb = BBox2d::new(p2d(-2, 3), v2d(1, 2));
        assert_eq!(bb2d(-2..-1, 3..5), bb);
    }

    #[test]
    fn test_from_point() {
        let bb = BBox2d::from_point(p2d(-2, 3));
        assert_eq!(bb2d(-2..=-2, 3..=3), bb);
    }

    #[test]
    fn test_from_points() {
        let bb = BBox2d::from_points(p2d(-2, 3), p2d(4, 7));
        assert_eq!(bb2d(-2..=4, 3..=7), bb);
    }

    #[test]
    fn test_min_2d() {
        let bb = bb2d(-2..=2, -1..=1);
        assert_eq!(p2d(-2, -1), bb.min());
    }
    #[test]
    fn test_min_3d() {
        let bb = bb3d(-2..=2, -1..=1, 1..=3);
        assert_eq!(p3d(-2, -1, 1), bb.min());
    }
    #[test]
    fn test_min_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(p4d(-2, -1, 1, -5), bb.min());
    }

    #[test]
    fn test_max_2d() {
        let bb = bb2d(-2..=2, -1..=1);
        assert_eq!(p2d(2, 1), bb.max());
    }
    #[test]
    fn test_max_3d() {
        let bb = bb3d(-2..=2, -1..=1, 1..=3);
        assert_eq!(p3d(2, 1, 3), bb.max());
    }
    #[test]
    fn test_max_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(p4d(2, 1, 3, -3), bb.max());
    }

    #[test]
    fn test_lengths_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(v2d(5, 3), bb.lengths());
    }
    #[test]
    fn test_lengths_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(v3d(5, 3, 3), bb.lengths());
    }
    #[test]
    fn test_lengths_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(v4d(5, 3, 3, 3), bb.lengths());
    }

    #[test]
    fn test_volume_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(15, bb.volume());
    }
    #[test]
    fn test_volume_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(45, bb.volume());
    }
    #[test]
    fn test_volume_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(135, bb.volume());
    }

    #[test]
    fn test_lub() {
        let bb0 = bb2d(-2..3, -1..2);
        let bb1 = bb2d(-5..4, 2..5);
        assert_eq!(bb2d(-5..4, -1..5), bb0.lub(&bb1));
    }

    #[test]
    fn test_extend_to() {
        let bb = bb2d(-2..3, -1..2);
        let p = p2d(3, 4);
        assert_eq!(bb2d(-2..4, -1..5), bb.extend_to(p));
    }

    #[test]
    fn test_x_start_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-2, bb.x_start());
    }
    #[test]
    fn test_x_start_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(-2, bb.x_start());
    }
    #[test]
    fn test_x_start_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(-2, bb.x_start());
    }

    #[test]
    fn test_x_end_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(3, bb.x_end());
    }
    #[test]
    fn test_x_end_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(3, bb.x_end());
    }
    #[test]
    fn test_x_end_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(3, bb.x_end());
    }

    #[test]
    fn test_x_min_2d() {
        let bb = bb2d(-2..=2, -1..=1);
        assert_eq!(-2, bb.x_min());
    }
    #[test]
    fn test_x_min_3d() {
        let bb = bb3d(-2..=2, -1..=1, 1..=3);
        assert_eq!(-2, bb.x_min());
    }
    #[test]
    fn test_x_min_4d() {
        let bb = bb4d(-2..=2, -1..=1, 1..=3, -5..=-3);
        assert_eq!(-2, bb.x_min());
    }

    #[test]
    fn test_x_max_2d() {
        let bb = bb2d(-2..=2, -1..=1);
        assert_eq!(2, bb.x_max());
    }
    #[test]
    fn test_x_max_3d() {
        let bb = bb3d(-2..=2, -1..=1, 1..=3);
        assert_eq!(2, bb.x_max());
    }
    #[test]
    fn test_x_max_4d() {
        let bb = bb4d(-2..=2, -1..=1, 1..=3, -5..=-3);
        assert_eq!(2, bb.x_max());
    }

    #[test]
    fn test_y_start_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-1, bb.y_start());
    }
    #[test]
    fn test_y_start_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(-1, bb.y_start());
    }
    #[test]
    fn test_y_start_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(-1, bb.y_start());
    }

    #[test]
    fn test_y_end_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(2, bb.y_end());
    }
    #[test]
    fn test_y_end_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(2, bb.y_end());
    }
    #[test]
    fn test_y_end_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(2, bb.y_end());
    }

    #[test]
    fn test_y_min_2d() {
        let bb = bb2d(-2..=2, -1..=1);
        assert_eq!(-1, bb.y_min());
    }
    #[test]
    fn test_y_min_3d() {
        let bb = bb3d(-2..=2, -1..=1, 1..=3);
        assert_eq!(-1, bb.y_min());
    }
    #[test]
    fn test_y_min_4d() {
        let bb = bb4d(-2..=2, -1..=1, 1..=3, -5..=-3);
        assert_eq!(-1, bb.y_min());
    }

    #[test]
    fn test_y_max_2d() {
        let bb = bb2d(-2..=2, -1..=1);
        assert_eq!(1, bb.y_max());
    }
    #[test]
    fn test_y_max_3d() {
        let bb = bb3d(-2..=2, -1..=1, 1..=3);
        assert_eq!(1, bb.y_max());
    }
    #[test]
    fn test_y_max_4d() {
        let bb = bb4d(-2..=2, -1..=1, 1..=3, -5..=-3);
        assert_eq!(1, bb.y_max());
    }

    #[test]
    fn test_z_start_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(1, bb.z_start());
    }
    #[test]
    fn test_z_start_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(1, bb.z_start());
    }

    #[test]
    fn test_z_end_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(4, bb.z_end());
    }
    #[test]
    fn test_z_end_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(4, bb.z_end());
    }

    #[test]
    fn test_z_min_3d() {
        let bb = bb3d(-2..=2, -1..=1, 1..=3);
        assert_eq!(1, bb.z_min());
    }
    #[test]
    fn test_z_min_4d() {
        let bb = bb4d(-2..=2, -1..=1, 1..=3, -5..=-3);
        assert_eq!(1, bb.z_min());
    }

    #[test]
    fn test_z_max_3d() {
        let bb = bb3d(-2..=2, -1..=1, 1..=3);
        assert_eq!(3, bb.z_max());
    }
    #[test]
    fn test_z_max_4d() {
        let bb = bb4d(-2..=2, -1..=1, 1..=3, -5..=-3);
        assert_eq!(3, bb.z_max());
    }

    #[test]
    fn test_w_start_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(-5, bb.w_start());
    }

    #[test]
    fn test_w_end_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(-2, bb.w_end());
    }

    #[test]
    fn test_w_min_4d() {
        let bb = bb4d(-2..=2, -1..=1, 1..=3, -5..=-3);
        assert_eq!(-5, bb.w_min());
    }

    #[test]
    fn test_w_max_4d() {
        let bb = bb4d(-2..=2, -1..=1, 1..=3, -5..=-3);
        assert_eq!(-3, bb.w_max());
    }

    #[test]
    fn test_x_len_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(5, bb.x_len());
    }
    #[test]
    fn test_x_len_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(5, bb.x_len());
    }
    #[test]
    fn test_x_len_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(5, bb.x_len());
    }

    #[test]
    fn test_y_len_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(3, bb.y_len());
    }
    #[test]
    fn test_y_len_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(3, bb.y_len());
    }
    #[test]
    fn test_y_len_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(3, bb.y_len());
    }

    #[test]
    fn test_z_len_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(3, bb.z_len());
    }
    #[test]
    fn test_z_len_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(3, bb.z_len());
    }

    #[test]
    fn test_w_len_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(3, bb.w_len());
    }

    #[test]
    fn test_area() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(15, bb.area());
    }

    #[test]
    fn test_contains() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(bb.contains(&p2d(0, 0)), true);
        assert_eq!(bb.contains(&p2d(-2, 0)), true);
        assert_eq!(bb.contains(&p2d(-3, 0)), false);
        assert_eq!(bb.contains(&p2d(2, 0)), true);
        assert_eq!(bb.contains(&p2d(3, 0)), false);
        assert_eq!(bb.contains(&p2d(0, -1)), true);
        assert_eq!(bb.contains(&p2d(0, -2)), false);
        assert_eq!(bb.contains(&p2d(0, 1)), true);
        assert_eq!(bb.contains(&p2d(0, 2)), false);
    }

    #[test]
    fn test_x_range_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-2..3, bb.x_range());
    }
    #[test]
    fn test_x_range_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(-2..3, bb.x_range());
    }
    #[test]
    fn test_x_range_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(-2..3, bb.x_range());
    }

    #[test]
    fn test_y_range_2d() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-1..2, bb.y_range());
    }
    #[test]
    fn test_y_range_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(-1..2, bb.y_range());
    }
    #[test]
    fn test_y_range_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(-1..2, bb.y_range());
    }

    #[test]
    fn test_z_range_3d() {
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(1..4, bb.z_range());
    }
    #[test]
    fn test_z_range_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(1..4, bb.z_range());
    }

    #[test]
    fn test_w_range_4d() {
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(-5..-2, bb.w_range());
    }

    #[test]
    fn test_center() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(p2d(0, 0), bb.center());
    }

    #[cfg(feature = "random")]
    #[test]
    fn test_random_point_2d() {
        let mut rng = StdRng::seed_from_u64(42);
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(p2d(-1, 0), bb.random_point(&mut rng));
    }
    #[cfg(feature = "random")]
    #[test]
    fn test_random_point_3d() {
        let mut rng = StdRng::seed_from_u64(42);
        let bb = bb3d(-2..3, -1..2, 1..4);
        assert_eq!(p3d(-1, 0, 3), bb.random_point(&mut rng));
    }
    #[cfg(feature = "random")]
    #[test]
    fn test_random_point_4d() {
        let mut rng = StdRng::seed_from_u64(42);
        let bb = bb4d(-2..3, -1..2, 1..4, -5..-2);
        assert_eq!(p4d(-1, 0, 3, -4), bb.random_point(&mut rng));
    }

    #[test]
    fn test_iter_2d() {
        let bb = bb2d(1..=2, -1..=0);
        let v = bb.iter().collect::<Vec<_>>();
        assert_eq!(vec![p2d(1, -1), p2d(2, -1), p2d(1, 0), p2d(2, 0)], v);
    }
    #[test]
    fn test_iter_3d() {
        let bb = bb3d(1..=2, -1..=0, -2..=-1);
        let v = bb.iter().collect::<Vec<_>>();
        assert_eq!(
            vec![
                p3d(1, -1, -2),
                p3d(2, -1, -2),
                p3d(1, 0, -2),
                p3d(2, 0, -2),
                p3d(1, -1, -1),
                p3d(2, -1, -1),
                p3d(1, 0, -1),
                p3d(2, 0, -1),
            ],
            v
        );
    }
    #[test]
    fn test_iter_4d() {
        let bb = bb4d(1..=2, -1..=0, -2..=-1, 3..=4);
        let v = bb.iter().collect::<Vec<_>>();
        assert_eq!(
            vec![
                p4d(1, -1, -2, 3),
                p4d(2, -1, -2, 3),
                p4d(1, 0, -2, 3),
                p4d(2, 0, -2, 3),
                p4d(1, -1, -1, 3),
                p4d(2, -1, -1, 3),
                p4d(1, 0, -1, 3),
                p4d(2, 0, -1, 3),
                p4d(1, -1, -2, 4),
                p4d(2, -1, -2, 4),
                p4d(1, 0, -2, 4),
                p4d(2, 0, -2, 4),
                p4d(1, -1, -1, 4),
                p4d(2, -1, -1, 4),
                p4d(1, 0, -1, 4),
                p4d(2, 0, -1, 4),
            ],
            v
        );
    }

    #[test]
    fn test_exact_size_iterator_2d() {
        let bb = bb2d(1..=2, -1..=0);
        let mut it = bb.into_iter();
        assert_eq!((4, Some(4)), it.size_hint());
        assert_eq!(4, it.len());
        let p = it.next();
        assert_eq!(Some(p2d(1, -1)), p);
        assert_eq!((3, Some(3)), it.size_hint());
        assert_eq!(3, it.len());
    }
    #[test]
    fn test_exact_size_iterator_3d() {
        let bb = bb3d(1..=2, -1..=0, -2..=-1);
        let mut it = bb.into_iter();
        assert_eq!((8, Some(8)), it.size_hint());
        assert_eq!(8, it.len());
        let p = it.next();
        assert_eq!(Some(p3d(1, -1, -2)), p);
        assert_eq!((7, Some(7)), it.size_hint());
        assert_eq!(7, it.len());
    }
    #[test]
    fn test_exact_size_iterator_4d() {
        let bb = bb4d(1..=2, -1..=0, -2..=-1, 3..=4);
        let mut it = bb.into_iter();
        assert_eq!((16, Some(16)), it.size_hint());
        assert_eq!(16, it.len());
        let p = it.next();
        assert_eq!(Some(p4d(1, -1, -2, 3)), p);
        assert_eq!((15, Some(15)), it.size_hint());
        assert_eq!(15, it.len());
    }
}
