//! n-dimensional bounding boxes.

use core::cmp::PartialOrd;
use core::convert::TryFrom;
use core::fmt::Debug;
use core::ops::Range;

#[cfg(feature = "random")]
use rand::distributions::uniform::SampleUniform;
#[cfg(feature = "random")]
use rand::Rng;

use crate::p2d;
use crate::Integer;
use crate::Point;
use crate::Point2d;
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
        let max = Point::with(|i| origin[i] + size[i] - S::from(1));
        assert!(max >= min);
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
        Point::<S, V>::from((self.min.to_vec() + self.max.to_vec()) / S::from(2))
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
    /// As always, lower bounds are inclusive, upper bounds exclusive.
    pub fn from_bounds(x_start: S, x_end: S, y_start: S, y_end: S) -> BBox2d<S> {
        assert!(x_start <= x_end && y_start <= y_end);
        let min = p2d(x_start, y_start);
        let max = p2d(x_end - S::from(1), y_end - S::from(1));
        BBox { min, max }
    }

    /// Returns the lower bound in the x-coordinate (inclusive).
    pub fn x_start(&self) -> S {
        self.min.x()
    }
    /// Returns the upper bound in the x-coordinate (exclusive).
    pub fn x_end(&self) -> S {
        self.max.x() + S::from(1)
    }
    /// Returns the lower bound in the x-coordinate (inclusive).
    pub fn x_min(&self) -> S {
        self.min.x()
    }
    /// Returns the upper bound in the x-coordinate (inclusive).
    pub fn x_max(&self) -> S {
        self.max.x()
    }
    /// Returns the lower bound in the x-coordinate (inclusive).
    #[deprecated = "Use `x_start` or `x_min` instead."]
    pub fn x0(&self) -> S {
        self.min.x()
    }
    /// Returns the upper bound in the x-coordinate (exclusive).
    #[deprecated = "Use `x_end` instead, or consider using `x_max`."]
    pub fn x1(&self) -> S {
        self.max.x() + S::from(1)
    }

    /// Returns the lower bound in the y-coordinate (inclusive).
    pub fn y_start(&self) -> S {
        self.min.y()
    }
    /// Returns the upper bound in the y-coordinate (exclusive).
    pub fn y_end(&self) -> S {
        self.max.y() + S::from(1)
    }
    /// Returns the lower bound in the y-coordinate (inclusive).
    pub fn y_min(&self) -> S {
        self.min.y()
    }
    /// Returns the upper bound in the y-coordinate (inclusive).
    pub fn y_max(&self) -> S {
        self.max.y()
    }
    /// Returns the lower bound in the y-coordinate (inclusive).
    #[deprecated = "Use `y_start` or `y_min` instead."]
    pub fn y0(&self) -> S {
        self.min.y()
    }
    /// Returns the upper bound in the y-coordinate (exclusive).
    #[deprecated = "Use `y_end` instead, or consider using `y_max`."]
    pub fn y1(&self) -> S {
        self.max.y() + S::from(1)
    }

    /// Returns the width of the bounding box.
    pub fn x_len(&self) -> usize
    where
        usize: TryFrom<S>,
    {
        usize::try_from(self.max.x() - self.min.x() + S::from(1)).unwrap_or(0)
    }

    /// Returns the height of the bounding box.
    pub fn y_len(&self) -> usize
    where
        usize: TryFrom<S>,
    {
        usize::try_from(self.max.y() - self.min.y() + S::from(1)).unwrap_or(0)
    }

    /// Returns the area of the bounding box.
    pub fn area(&self) -> usize
    where
        usize: TryFrom<S>,
    {
        self.x_len() * self.y_len()
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
    pub fn iter(&self) -> Iter<S>
    where
        usize: TryFrom<S>,
    {
        let next_point = Some(self.min);
        Iter {
            bbox: self,
            next_point,
        }
    }

    /// Returns the sequential index for a given point.
    ///
    /// Points are counted by row.
    pub fn seq_index(&self, p: Point2d<S>) -> usize
    where
        usize: TryFrom<S>,
    {
        assert!(self.contains(&p));
        let dx = usize::try_from(p.x() - self.x_start()).unwrap_or(0);
        let dy = usize::try_from(p.y() - self.y_start()).unwrap_or(0);
        let w = self.x_len();
        dx + dy * w
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

/// Creates a 2d bounding box from ranges of x and y coordinates.
pub fn bb2d<S: Integer>(x_range: Range<S>, y_range: Range<S>) -> BBox2d<S> {
    let Range {
        start: x_start,
        end: x_end,
    } = x_range;
    let Range {
        start: y_start,
        end: y_end,
    } = y_range;
    BBox2d::from_bounds(x_start, x_end, y_start, y_end)
}

impl<'a, S: Integer> IntoIterator for &'a BBox2d<S>
where
    usize: TryFrom<S>,
{
    type Item = Point2d<S>;

    type IntoIter = Iter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
/// An iterator over the points in a 2d bounding box.
pub struct Iter<'a, S: Integer> {
    bbox: &'a BBox2d<S>,
    next_point: Option<Point2d<S>>,
}

impl<'a, S: Integer> Iterator for Iter<'a, S>
where
    usize: TryFrom<S>,
{
    type Item = Point2d<S>;

    fn next(&mut self) -> Option<Point2d<S>> {
        match self.next_point {
            Some(p) => {
                // Move to next point
                let new_x = p.x() + S::from(1);
                self.next_point = {
                    if new_x < self.bbox.x_end() {
                        Some(p2d(new_x, p.y()))
                    } else {
                        let new_y = p.y() + S::from(1);
                        if new_y < self.bbox.y_end() {
                            Some(p2d(self.bbox.x_start(), new_y))
                        } else {
                            None
                        }
                    }
                };
                Some(p)
            }
            None => None,
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = {
            match self.next_point {
                None => 0,
                Some(p) => self.bbox.area() - self.bbox.seq_index(p),
            }
        };
        (len, Some(len))
    }
}

impl<'a, S: Integer> ExactSizeIterator for Iter<'a, S>
where
    S: Copy + From<u8> + PartialOrd,
    usize: TryFrom<S>,
{
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

#[cfg(test)]
mod tests {
    use crate::bb2d;
    use crate::p2d;
    use crate::v2d;
    use crate::BBox2d;

    #[test]
    fn test_new() {
        let bb = BBox2d::new(p2d(-2, 3), v2d(1, 2));
        assert_eq!(bb2d(-2..-1, 3..5), bb);
    }

    #[test]
    fn test_from_point() {
        let bb = BBox2d::from_point(p2d(-2, 3));
        assert_eq!(bb2d(-2..-1, 3..4), bb);
    }

    #[test]
    fn test_from_points() {
        let bb = BBox2d::from_points(p2d(-2, 3), p2d(4, 7));
        assert_eq!(bb2d(-2..5, 3..8), bb);
    }

    #[test]
    fn test_min() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(p2d(-2, -1), bb.min());
    }
    #[test]
    fn test_max() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(p2d(2, 1), bb.max());
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
    fn test_x_start() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-2, bb.x_start());
    }
    #[test]
    fn test_x_end() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(3, bb.x_end());
    }
    #[test]
    fn test_x_min() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-2, bb.x_min());
    }
    #[test]
    fn test_x_max() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(2, bb.x_max());
    }
    #[test]
    #[allow(deprecated)]
    fn test_x0() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-2, bb.x0());
    }
    #[test]
    #[allow(deprecated)]
    fn test_x1() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(3, bb.x1());
    }

    #[test]
    fn test_y_start() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-1, bb.y_start());
    }
    #[test]
    fn test_y_end() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(2, bb.y_end());
    }
    #[test]
    fn test_y_min() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-1, bb.y_min());
    }
    #[test]
    fn test_y_max() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(1, bb.y_max());
    }
    #[test]
    #[allow(deprecated)]
    fn test_y0() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-1, bb.y0());
    }
    #[test]
    #[allow(deprecated)]
    fn test_y1() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(2, bb.y1());
    }

    #[test]
    fn test_x_len() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(5, bb.x_len());
    }
    #[test]
    fn test_y_len() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(3, bb.y_len());
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
    fn test_x_range() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-2..3, bb.x_range());
    }
    #[test]
    fn test_y_range() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(-1..2, bb.y_range());
    }

    #[test]
    fn test_center() {
        let bb = bb2d(-2..3, -1..2);
        assert_eq!(p2d(0, 0), bb.center());
    }

    #[test]
    fn test_iter() {
        let bb = bb2d(1..3, -1..1);
        for p in bb.iter() {
            println!("{:?}", p);
        }
        let v = bb.iter().collect::<Vec<_>>();
        assert_eq!(vec![p2d(1, -1), p2d(2, -1), p2d(1, 0), p2d(2, 0)], v);
    }
    #[test]
    fn test_exact_size_iterator() {
        let bb = bb2d(1..3, -1..1);
        let mut it = bb.into_iter();
        assert_eq!((4, Some(4)), it.size_hint());
        assert_eq!(4, it.len());
        let p = it.next();
        assert_eq!(Some(p2d(1, -1)), p);
        assert_eq!((3, Some(3)), it.size_hint());
        assert_eq!(3, it.len());
    }
}
