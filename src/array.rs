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
pub struct Array2d<S: Integer, T>
where
    usize: TryFrom<S>,
{
    bounds: BBox2d<S>,
    data: Vec<T>,
}

impl<S: Integer, T> Array2d<S, T>
where
    usize: TryFrom<S>,
{
    /// Creates a new array with the given bounds
    /// that is filled with copies of a given element.
    pub fn new(bounds: BBox2d<S>, d: T) -> Array2d<S, T>
    where
        T: Clone,
    {
        let data = repeat(d).take(bounds.area() as usize).collect::<Vec<_>>();
        Array2d { bounds, data }
    }

    /// Creates a new array with the given bounds
    /// that is filled using a function which takes a location as input.
    pub fn with<F>(bounds: BBox2d<S>, f: F) -> Array2d<S, T>
    where
        F: Fn(Point2d<S>) -> T,
    {
        let data = bounds.iter().map(f).collect::<Vec<_>>();
        Array2d { bounds, data }
    }

    /// Creates a new array with the given bounds
    /// that is filled using a function which takes a location as input.
    #[deprecated = "Use `with` instead."]
    pub fn new_with<F>(bounds: BBox2d<S>, f: F) -> Array2d<S, T>
    where
        F: Fn(Point2d<S>) -> T,
    {
        Self::with(bounds, f)
    }

    /// Creates a new array with the given bounds
    /// that is filled using a function which takes a location as input.
    pub fn from_vec(v: Vec<Vec<T>>) -> Array2d<S, T>
    where
        T: Copy,
        S: TryFrom<usize>,
    {
        let y_len = S::try_from(v.len()).unwrap_or(0.into());
        let x_len = if y_len == 0.into() {
            0.into()
        } else {
            S::try_from(v[0].len()).unwrap_or(0.into())
        };
        let bounds = BBox2d::from_bounds(0.into(), x_len, 0.into(), y_len);
        Array2d::with(bounds, |p| {
            let x = usize::try_from(p.x()).unwrap_or(0_usize);
            let y = usize::try_from(p.y()).unwrap_or(0_usize);
            v[y][x]
        })
    }

    /// Returns the bounds of the array.
    pub fn bounds(&self) -> BBox2d<S> {
        self.bounds
    }

    /// Returns a reference to the element at the index
    /// or None if the index is out of bounds.
    pub fn get(&self, index: Point2d<S>) -> Option<&T> {
        if self.bounds().contains(index) {
            Some(&self.data[self.bounds.seq_index(index)])
        } else {
            None
        }
    }

    /// Returns a reference to the element at the index
    /// or None if the index is out of bounds.
    pub fn get_mut(&mut self, index: Point2d<S>) -> Option<&mut T> {
        if self.bounds().contains(index) {
            Some(&mut self.data[self.bounds.seq_index(index)])
        } else {
            None
        }
    }

    /// Returns an iterator over the elements of the array.
    pub fn iter(&self) -> impl Iterator<Item = &T>
    where
        usize: TryFrom<S>,
    {
        Array2dIter {
            array: self,
            iter: self.bounds.iter(),
        }
    }
}

struct Array2dIter<'a, S: Integer, T, Iter>
where
    usize: TryFrom<S>,
    Iter: Iterator<Item = Point2d<S>>,
{
    array: &'a Array2d<S, T>,
    iter: Iter,
}
impl<'a, S: Integer, T, Iter> Iterator for Array2dIter<'a, S, T, Iter>
where
    usize: TryFrom<S>,
    Iter: Iterator<Item = Point2d<S>>,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(p) = self.iter.next() {
            Some(&self.array[p])
        } else {
            None
        }
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
    fn test_from_vec() {
        let v = vec![vec!['0', '1'], vec!['2', '3']];
        let a = Array2d::from_vec(v);
        assert_eq!(a[p2d(0, 0)], '0');
        assert_eq!(a[p2d(1, 0)], '1');
        assert_eq!(a[p2d(0, 1)], '2');
        assert_eq!(a[p2d(1, 1)], '3');
    }

    #[test]
    fn test_get() {
        let r = bb2d(-2..3, -1..2);
        let a = Array2d::with(r, |p| if p == p2d(0, 0) { '*' } else { '.' });
        assert_eq!(a.get(p2d(0, 0)), Some(&'*'));
        assert_eq!(a.get(p2d(1, 1)), Some(&'.'));
        assert_eq!(a.get(p2d(3, 2)), None);
    }

    #[test]
    fn test_get_mut() {
        let r = bb2d(-2..3, -1..2);
        let mut a = Array2d::new(r, '.');
        *a.get_mut(p2d(0, 0)).unwrap() = '*';
        assert_eq!(a[p2d(0, 0)], '*');
        assert_eq!(a.get(p2d(1, 1)), Some(&'.'));
        assert_eq!(a.get_mut(p2d(3, 2)), None);
    }

    #[test]
    fn test_index() {
        let r = bb2d(-2..3, -1..2);
        let a = Array2d::with(r, |p| if p == p2d(0, 0) { '*' } else { '.' });
        assert_eq!(a[p2d(0, 0)], '*');
        assert_eq!(a[p2d(1, 1)], '.');
    }

    #[test]
    fn test_index_mut() {
        let r = bb2d(-2..3, -1..2);
        let mut a = Array2d::new(r, '.');
        a[p2d(0, 0)] = '*';
        assert_eq!(a[p2d(0, 0)], '*');
        assert_eq!(a[p2d(-2, -1)], '.');
        assert_eq!(a[p2d(-2, 1)], '.');
        assert_eq!(a[p2d(2, -1)], '.');
        assert_eq!(a[p2d(2, 1)], '.');
    }

    #[test]
    fn test_iter() {
        let r = bb2d(0..2, 0..2);
        let a = Array2d::with(r, |p| p.x() + p.y());
        let v = a.iter().cloned().collect::<Vec<_>>();
        assert_eq!(v, vec![0, 1, 1, 2]);
    }
}
