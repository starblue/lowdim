//! n-dimensional arrays indexed by points in a bounding box.

use core::fmt;
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
    bbox: BBox2d<S>,
    data: Box<[T]>,
}

impl<S: Integer, T> Array2d<S, T>
where
    usize: TryFrom<S>,
{
    /// Creates a new array with the given bounding box
    /// that is filled with copies of a given element.
    pub fn new(bbox: BBox2d<S>, d: T) -> Array2d<S, T>
    where
        T: Clone,
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        let data = repeat(d).take(bbox.area()).collect::<Box<[_]>>();
        Array2d { bbox, data }
    }

    /// Creates a new array with the given bounding box
    /// that is filled using a function which takes a location as input.
    pub fn with<F>(bbox: BBox2d<S>, f: F) -> Array2d<S, T>
    where
        F: Fn(Point2d<S>) -> T,
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        let data = bbox.iter().map(f).collect::<Box<[_]>>();
        Array2d { bbox, data }
    }

    /// Creates a new array with the given bounding box
    /// that is filled from a vector of vectors.
    pub fn from_vec(v: Vec<Vec<T>>) -> Array2d<S, T>
    where
        T: Copy,
        S: TryFrom<usize>,
        <S as TryFrom<usize>>::Error: fmt::Debug,
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        let y_len = S::try_from(v.len()).unwrap();
        let x_len = if y_len == S::zero() {
            S::zero()
        } else {
            S::try_from(v[0].len()).unwrap()
        };
        let bbox = BBox2d::from_bounds(S::zero(), x_len, S::zero(), y_len);
        Array2d::with(bbox, |p| {
            let x = usize::try_from(p.x()).unwrap();
            let y = usize::try_from(p.y()).unwrap();
            v[y][x]
        })
    }

    /// Returns the bounding box of the array.
    pub fn bbox(&self) -> BBox2d<S> {
        self.bbox
    }
    /// Returns the bounding box of the array.
    #[deprecated = "Use `bbox` instead."]
    pub fn bounds(&self) -> BBox2d<S> {
        self.bbox
    }

    /// Returns a reference to the element at the index
    /// or None if the index is out of bounds.
    pub fn get(&self, index: Point2d<S>) -> Option<&T>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        if self.bbox().contains(&index) {
            Some(&self.data[self.bbox.seq_index(index)])
        } else {
            None
        }
    }

    /// Returns a reference to the element at the index
    /// or None if the index is out of bounds.
    pub fn get_mut(&mut self, index: Point2d<S>) -> Option<&mut T>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        if self.bbox().contains(&index) {
            Some(&mut self.data[self.bbox.seq_index(index)])
        } else {
            None
        }
    }

    /// Returns an iterator over the elements of the array.
    pub fn iter(&self) -> impl Iterator<Item = &T>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        Array2dIter {
            array: self,
            iter: self.bbox.iter(),
        }
    }
}

struct Array2dIter<'a, S: Integer, T, Iter>
where
    Iter: Iterator<Item = Point2d<S>>,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    array: &'a Array2d<S, T>,
    iter: Iter,
}
impl<'a, S: Integer, T, Iter> Iterator for Array2dIter<'a, S, T, Iter>
where
    Iter: Iterator<Item = Point2d<S>>,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
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
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    type Output = T;

    fn index(&self, index: Point2d<S>) -> &T {
        &self.data[self.bbox.seq_index(index)]
    }
}

impl<S: Integer, T> ops::IndexMut<Point2d<S>> for Array2d<S, T>
where
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    fn index_mut(&mut self, index: Point2d<S>) -> &mut T {
        &mut self.data[self.bbox.seq_index(index)]
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
