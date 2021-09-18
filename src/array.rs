//! n-dimensional arrays indexed by points in a bounding box.

use core::ops;

use std::convert::TryFrom;
use std::iter::repeat;

use crate::BBox2d;
use crate::Integer;
use crate::Point2d;

/// A two-dimensional array indexed by points in a bounding box.
///
/// The starting index and size is given by a rectangle,
/// i.e. x- and y-index don't need to start at zero.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Array2d<S: Integer, T> {
    bounds: BBox2d<S>,
    data: Vec<T>,
}

impl<S: Integer, T> Array2d<S, T> {
    /// Creates a new array with the given bounds
    /// that is filled with copies of a given element.
    pub fn new(bounds: BBox2d<S>, d: T) -> Array2d<S, T>
    where
        usize: TryFrom<S>,
        T: Clone,
    {
        let data = repeat(d).take(bounds.area() as usize).collect::<Vec<_>>();
        Array2d { bounds, data }
    }

    /// Creates a new array with the given bounds
    /// that is filled using a function which takes a location as input.
    pub fn new_with<F>(bounds: BBox2d<S>, f: F) -> Array2d<S, T>
    where
        F: Fn(Point2d<S>) -> T,
        usize: TryFrom<S>,
    {
        let data = bounds.iter().map(f).collect::<Vec<_>>();
        Array2d { bounds, data }
    }

    /// Creates a new array with the given bounds
    /// that is filled using a function which takes a location as input.
    pub fn from_vec(v: Vec<Vec<T>>) -> Array2d<S, T>
    where
        T: Copy,
        S: TryFrom<usize>,
        usize: TryFrom<S>,
    {
        let y_len = S::try_from(v.len()).unwrap_or(0.into());
        let x_len = if y_len == 0.into() {
            0.into()
        } else {
            S::try_from(v[0].len()).unwrap_or(0.into())
        };
        let bounds = BBox2d::from_bounds(0.into(), x_len, 0.into(), y_len);
        Array2d::new_with(bounds, |p| {
            let x = usize::try_from(p.x()).unwrap_or(0_usize);
            let y = usize::try_from(p.y()).unwrap_or(0_usize);
            v[y][x]
        })
    }

    /// Returns the bounds of the array.
    pub fn bounds(&self) -> BBox2d<S> {
        self.bounds
    }
}

impl<S: Integer, T> ops::Index<Point2d<S>> for Array2d<S, T>
where
    usize: TryFrom<S>,
{
    type Output = T;

    fn index(&self, index: Point2d<S>) -> &T {
        &self.data[self.bounds.seq_index(index)]
    }
}

impl<S: Integer, T> ops::IndexMut<Point2d<S>> for Array2d<S, T>
where
    usize: TryFrom<S>,
{
    fn index_mut(&mut self, index: Point2d<S>) -> &mut T {
        &mut self.data[self.bounds.seq_index(index)]
    }
}

#[cfg(test)]
mod tests {
    use crate::bb2d;
    use crate::p2d;
    use crate::Array2d;

    #[test]
    fn test_index() {
        let r = bb2d(-2..3, -1..2);
        let mut a = Array2d::new_with(r, |p| if p == p2d(0, 0) { '*' } else { '.' });
        assert_eq!(a[p2d(0, 0)], '*');
        a[p2d(1, 1)] = '+';
        assert_eq!(a[p2d(1, 1)], '+');
    }
}
