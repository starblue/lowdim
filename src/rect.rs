use core::cmp::PartialOrd;
use core::convert::TryFrom;
use core::fmt::Debug;
use core::ops::Range;

use rand::distributions::uniform::SampleUniform;
use rand::Rng;

use crate::p2d;
use crate::v2d;
use crate::Integer;
use crate::Point2d;
use crate::Vec2d;

/// A rectangle in the discrete plane.
///
/// The coordinates of the size must be non-negative.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Rect2d<S: Integer> {
    origin: Point2d<S>,
    size: Vec2d<S>,
}

/// A two-dimensional rectangle.
impl<S: Integer> Rect2d<S> {
    /// Constructs a rectangle from origin and size.
    pub fn new(origin: Point2d<S>, size: Vec2d<S>) -> Rect2d<S> {
        assert!(size.x() >= 0.into() && size.y() >= 0.into());
        Rect2d { origin, size }
    }
    /// Constructs a rectangle from bounds.
    ///
    /// As always, lower bounds are inclusive, upper bounds exclusive.
    pub fn from_bounds(x0: S, x1: S, y0: S, y1: S) -> Rect2d<S> {
        assert!(x0 <= x1 && y0 <= y1);
        let origin = p2d(x0, y0);
        let size = v2d(x1 - x0, y1 - y0);
        Rect2d::new(origin, size)
    }

    /// Returns the lower bound in the x-coordinate (inclusive).
    pub fn x0(&self) -> S {
        self.origin.x()
    }
    /// Returns the upper bound in the x-coordinate (exclusive).
    pub fn x1(&self) -> S {
        self.origin.x() + self.size.x()
    }

    /// Returns the lower bound in the y-coordinate (inclusive).
    pub fn y0(&self) -> S {
        self.origin.y()
    }
    /// Returns the upper bound in the y-coordinate (exclusive).
    pub fn y1(&self) -> S {
        self.origin.y() + self.size.y()
    }

    /// Returns the width of the rectangle.
    pub fn width(&self) -> usize
    where
        usize: TryFrom<S>,
    {
        usize::try_from(self.size.x()).unwrap_or(0)
    }
    /// Returns the height of the rectangle.
    pub fn height(&self) -> usize
    where
        usize: TryFrom<S>,
    {
        usize::try_from(self.size.y()).unwrap_or(0)
    }

    /// Returns the area of the rectangle.
    pub fn area(&self) -> usize
    where
        usize: TryFrom<S>,
    {
        self.width() * self.height()
    }

    /// Returns true if the point is inside the rectangle.
    pub fn contains(&self, p: Point2d<S>) -> bool {
        self.x0() <= p.x() && p.x() < self.x1() && self.y0() <= p.y() && p.y() < self.y1()
    }

    pub fn is_empty(&self) -> bool
    where
        usize: TryFrom<S>,
    {
        self.width() == 0 || self.height() == 0
    }

    /// The range of the x coordinate
    pub fn x_range(&self) -> Range<S> {
        self.x0()..self.x1()
    }
    /// The range of the y coordinate
    pub fn y_range(&self) -> Range<S> {
        self.y0()..self.y1()
    }

    /// Returns an iterator over the points in the rectangle.
    ///
    /// Points are returned by row.
    pub fn iter(&self) -> Iter<S>
    where
        usize: TryFrom<S>,
    {
        let next_point = {
            if self.is_empty() {
                None
            } else {
                Some(self.origin)
            }
        };
        Iter {
            rect: &self,
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
        assert!(self.contains(p));
        let dx = usize::try_from(p.x() - self.x0()).unwrap_or(0);
        let dy = usize::try_from(p.y() - self.y0()).unwrap_or(0);
        let w = self.width();
        dx + dy * w
    }

    /// Returns a random point inside the rectangle.
    ///
    /// Uses a uniform distribution.
    /// TODO make dependency on rand configurable
    pub fn random_point<R>(&self, rng: &mut R) -> Point2d<S>
    where
        R: Rng,
        S: SampleUniform,
    {
        let x = rng.gen_range(self.x0(), self.x1());
        let y = rng.gen_range(self.y0(), self.y1());
        p2d(x, y)
    }
}

pub fn r2d<S: Integer>(x0: S, x1: S, y0: S, y1: S) -> Rect2d<S> {
    Rect2d::from_bounds(x0, x1, y0, y1)
}

impl<'a, S: Integer> IntoIterator for &'a Rect2d<S>
where
    usize: TryFrom<S>,
{
    type Item = Point2d<S>;

    type IntoIter = Iter<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
pub struct Iter<'a, S: Integer> {
    rect: &'a Rect2d<S>,
    next_point: Option<Point2d<S>>,
}

impl<'a, S: Integer> Iterator for Iter<'a, S> {
    type Item = Point2d<S>;

    fn next(&mut self) -> Option<Point2d<S>> {
        match self.next_point {
            Some(p) => {
                // Move to next point
                let new_x = p.x() + S::from(1);
                self.next_point = {
                    if new_x < self.rect.x1() {
                        Some(p2d(new_x, p.y()))
                    } else {
                        let new_y = p.y() + S::from(1);
                        if new_y < self.rect.y1() {
                            Some(p2d(self.rect.x0(), new_y))
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
}

impl<'a, S: Integer> ExactSizeIterator for Iter<'a, S>
where
    S: Copy + From<u8> + PartialOrd,
    usize: TryFrom<S>,
{
    fn len(&self) -> usize {
        match self.next_point {
            None => 0,
            Some(p) => self.rect.area() - self.rect.seq_index(p),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::p2d;
    use crate::r2d;

    #[test]
    fn test_accessors() {
        let r = r2d(-2, 3, -1, 2);
        assert_eq!(r.x0(), -2);
        assert_eq!(r.x1(), 3);
        assert_eq!(r.y0(), -1);
        assert_eq!(r.y1(), 2);
        assert_eq!(r.width(), 5);
        assert_eq!(r.height(), 3);
        assert_eq!(r.area(), 15);
    }

    #[test]
    fn test_contains() {
        let r = r2d(-2, 3, -1, 2);
        assert_eq!(r.contains(p2d(0, 0)), true);
        assert_eq!(r.contains(p2d(-2, 0)), true);
        assert_eq!(r.contains(p2d(-3, 0)), false);
        assert_eq!(r.contains(p2d(2, 0)), true);
        assert_eq!(r.contains(p2d(3, 0)), false);
        assert_eq!(r.contains(p2d(0, -1)), true);
        assert_eq!(r.contains(p2d(0, -2)), false);
        assert_eq!(r.contains(p2d(0, 1)), true);
        assert_eq!(r.contains(p2d(0, 2)), false);
    }

    #[test]
    fn test_iter() {
        let r = r2d(1, 3, -1, 1);
        for p in r.iter() {
            println!("{:?}", p);
        }
        let v = r.iter().collect::<Vec<_>>();
        assert_eq!(v, vec![p2d(1, -1), p2d(2, -1), p2d(1, 0), p2d(2, 0)]);
    }
}
