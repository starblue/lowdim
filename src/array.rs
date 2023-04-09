//! n-dimensional arrays indexed by points in a bounding box.

use core::marker::PhantomData;

use core::fmt;
use core::ops;

use std::iter::repeat;

use crate::bb2d;
use crate::BBox;
use crate::Integer;
use crate::Layout;
use crate::Point;
use crate::Vec2d;
use crate::Vec3d;
use crate::Vec4d;
use crate::Vector;

/// An array indexed by points in a bounding box.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Array<S, V, T, L = <V as Vector<S>>::DefaultLayout>
where
    S: Integer,
    V: Vector<S>,
    L: Layout<S, V>,
{
    _s: PhantomData<S>,
    _v: PhantomData<V>,
    data: Box<[T]>,
    layout: L,
}
impl<S, V, T> Array<S, V, T, <V as Vector<S>>::DefaultLayout>
where
    S: Integer,
    V: Vector<S>,
{
    /// Creates a new array with the given bounding box
    /// that is filled with copies of a given element.
    pub fn new(bbox: BBox<S, V>, d: T) -> Self
    where
        T: Clone,
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
        <S as TryFrom<usize>>::Error: fmt::Debug,
    {
        let layout = <<V as Vector<S>>::DefaultLayout>::new(bbox);
        let data = repeat(d).take(bbox.volume()).collect::<Box<[_]>>();
        Array {
            _s: PhantomData,
            _v: PhantomData,
            layout,
            data,
        }
    }

    /// Creates a new array with the given bounding box
    /// that is filled using a function which takes a location as input.
    pub fn with<F>(bbox: BBox<S, V>, f: F) -> Self
    where
        F: FnMut(Point<S, V>) -> T,
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
        <S as TryFrom<usize>>::Error: fmt::Debug,
    {
        let layout = <<V as Vector<S>>::DefaultLayout>::new(bbox);
        let data = layout.points().map(f).collect::<Box<[_]>>();
        Array {
            _s: PhantomData,
            _v: PhantomData,
            layout,
            data,
        }
    }
}
impl<S, V, T, L> Array<S, V, T, L>
where
    S: Integer,
    V: Vector<S>,
    L: Layout<S, V>,
{
    /// Returns the bounding box of the array.
    pub fn bbox(&self) -> BBox<S, V> {
        self.layout.bbox()
    }
    /// Returns the bounding box of the array.
    #[deprecated = "Use `bbox` instead."]
    pub fn bounds(&self) -> BBox<S, V> {
        self.bbox()
    }

    /// Returns a reference to the element at the index
    /// or None if the index is out of bounds.
    pub fn get(&self, index: Point<S, V>) -> Option<&T>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        if let Some(i) = self.layout.index(index) {
            Some(&self.data[i])
        } else {
            None
        }
    }

    /// Returns a reference to the element at the index
    /// or None if the index is out of bounds.
    pub fn get_mut(&mut self, index: Point<S, V>) -> Option<&mut T>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        if let Some(i) = self.layout.index(index) {
            Some(&mut self.data[i])
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
        ArrayIter {
            array: self,
            iter: self.layout.points(),
        }
    }
}

/// A 2d array indexed by points in a bounding box.
///
/// The starting index and size is given by a rectangle,
/// i.e. x- and y-index don't need to start at zero.
pub type Array2d<S, T, L = <Vec2d<S> as Vector<S>>::DefaultLayout> = Array<S, Vec2d<S>, T, L>;
/// A 3d array indexed by points in a bounding box.
pub type Array3d<S, T, L> = Array<S, Vec3d<S>, T, L>;
/// A 4d array indexed by points in a bounding box.
pub type Array4d<S, T, L> = Array<S, Vec4d<S>, T, L>;

impl<S, T> Array2d<S, T, <Vec2d<S> as Vector<S>>::DefaultLayout>
where
    S: Integer,
{
    /// Creates a new array with the given bounds
    /// that is filled from a vector of vectors.
    pub fn from_vec(v: Vec<Vec<T>>) -> Self
    where
        T: Copy,
        <S as TryFrom<usize>>::Error: fmt::Debug,
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        let y_len = S::from_usize(v.len());
        let x_len = if y_len == S::zero() {
            S::zero()
        } else {
            S::try_from(v[0].len()).unwrap()
        };
        let bbox = bb2d(S::zero()..x_len, S::zero()..y_len);
        Array2d::with(bbox, |p| {
            let x = p.x().to_usize();
            let y = p.y().to_usize();
            v[y][x]
        })
    }
}

struct ArrayIter<'a, S, V, T, Iter, L>
where
    S: Integer,
    V: Vector<S>,
    Iter: Iterator<Item = Point<S, V>>,
    L: Layout<S, V>,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    array: &'a Array<S, V, T, L>,
    iter: Iter,
}
impl<'a, S, V, T, Iter, L> Iterator for ArrayIter<'a, S, V, T, Iter, L>
where
    S: Integer,
    V: Vector<S>,
    Iter: Iterator<Item = Point<S, V>>,
    L: Layout<S, V>,
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

impl<S, V, T, L> ops::Index<Point<S, V>> for Array<S, V, T, L>
where
    S: Integer,
    V: Vector<S>,
    L: Layout<S, V>,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    type Output = T;

    fn index(&self, index: Point<S, V>) -> &T {
        self.get(index).unwrap()
    }
}

impl<S, V, T, L> ops::IndexMut<Point<S, V>> for Array<S, V, T, L>
where
    S: Integer,
    V: Vector<S>,
    L: Layout<S, V>,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    fn index_mut(&mut self, index: Point<S, V>) -> &mut T {
        self.get_mut(index).unwrap()
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
    use crate::Array2d;
    use crate::Array3d;
    use crate::Array4d;

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
    fn test_get_2d() {
        let r = bb2d(-2..3, -1..2);
        let a = Array2d::with(r, |p| if p == p2d(0, 0) { '*' } else { '.' });
        assert_eq!(a.get(p2d(0, 0)), Some(&'*'));
        assert_eq!(a.get(p2d(1, 1)), Some(&'.'));
        assert_eq!(a.get(p2d(3, 2)), None);
    }
    #[test]
    fn test_get_3d() {
        let r = bb3d(-2..3, -1..2, 0..2);
        let a = Array3d::with(r, |p| if p == p3d(0, 0, 0) { '*' } else { '.' });
        assert_eq!(a.get(p3d(0, 0, 0)), Some(&'*'));
        assert_eq!(a.get(p3d(1, 1, 1)), Some(&'.'));
        assert_eq!(a.get(p3d(3, 2, 0)), None);
    }
    #[test]
    fn test_get_4d() {
        let r = bb4d(-2..3, -1..2, 0..2, 1..3);
        let a = Array4d::with(r, |p| if p == p4d(0, 0, 0, 1) { '*' } else { '.' });
        assert_eq!(a.get(p4d(0, 0, 0, 1)), Some(&'*'));
        assert_eq!(a.get(p4d(1, 1, 1, 1)), Some(&'.'));
        assert_eq!(a.get(p4d(3, 2, 0, 1)), None);
    }

    #[test]
    fn test_get_mut_2d() {
        let r = bb2d(-2..3, -1..2);
        let mut a = Array2d::new(r, '.');
        *a.get_mut(p2d(0, 0)).unwrap() = '*';
        assert_eq!(a[p2d(0, 0)], '*');
        assert_eq!(a.get(p2d(1, 1)), Some(&'.'));
        assert_eq!(a.get_mut(p2d(3, 2)), None);
    }
    #[test]
    fn test_get_mut_3d() {
        let r = bb3d(-2..3, -1..2, 0..2);
        let mut a = Array3d::new(r, '.');
        *a.get_mut(p3d(0, 0, 0)).unwrap() = '*';
        assert_eq!(a[p3d(0, 0, 0)], '*');
        assert_eq!(a.get(p3d(1, 1, 1)), Some(&'.'));
        assert_eq!(a.get_mut(p3d(3, 2, 0)), None);
    }
    #[test]
    fn test_get_mut_4d() {
        let r = bb4d(-2..3, -1..2, 0..2, 1..3);
        let mut a = Array4d::new(r, '.');
        *a.get_mut(p4d(0, 0, 0, 1)).unwrap() = '*';
        assert_eq!(a[p4d(0, 0, 0, 1)], '*');
        assert_eq!(a.get(p4d(1, 1, 1, 1)), Some(&'.'));
        assert_eq!(a.get_mut(p4d(3, 2, 0, 1)), None);
    }

    #[test]
    fn test_index_2d() {
        let r = bb2d(-2..3, -1..2);
        let a = Array2d::with(r, |p| if p == p2d(0, 0) { '*' } else { '.' });
        assert_eq!(a[p2d(0, 0)], '*');
        assert_eq!(a[p2d(1, 1)], '.');
    }
    #[test]
    fn test_index_3d() {
        let r = bb3d(-2..3, -1..2, 0..2);
        let a = Array3d::with(r, |p| if p == p3d(0, 0, 0) { '*' } else { '.' });
        assert_eq!(a[p3d(0, 0, 0)], '*');
        assert_eq!(a[p3d(1, 1, 1)], '.');
    }
    #[test]
    fn test_index_4d() {
        let r = bb4d(-2..3, -1..2, 0..2, 1..3);
        let a = Array4d::with(r, |p| if p == p4d(0, 0, 0, 1) { '*' } else { '.' });
        assert_eq!(a[p4d(0, 0, 0, 1)], '*');
        assert_eq!(a[p4d(1, 1, 1, 1)], '.');
    }

    #[test]
    fn test_index_mut_2d() {
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
    fn test_index_mut_3d() {
        let r = bb3d(-2..3, -1..2, 0..2);
        let mut a = Array3d::new(r, '.');
        a[p3d(0, 0, 0)] = '*';
        assert_eq!(a[p3d(0, 0, 0)], '*');
        assert_eq!(a[p3d(-2, -1, 0)], '.');
        assert_eq!(a[p3d(-2, 1, 0)], '.');
        assert_eq!(a[p3d(2, -1, 0)], '.');
        assert_eq!(a[p3d(2, 1, 0)], '.');
        assert_eq!(a[p3d(-2, -1, 1)], '.');
        assert_eq!(a[p3d(-2, 1, 1)], '.');
        assert_eq!(a[p3d(2, -1, 1)], '.');
        assert_eq!(a[p3d(2, 1, 1)], '.');
    }
    #[test]
    fn test_index_mut_4d() {
        let r = bb4d(-2..3, -1..2, 0..2, 1..3);
        let mut a = Array4d::new(r, '.');
        a[p4d(0, 0, 0, 1)] = '*';
        assert_eq!(a[p4d(0, 0, 0, 1)], '*');
        assert_eq!(a[p4d(-2, -1, 0, 1)], '.');
        assert_eq!(a[p4d(-2, 1, 0, 1)], '.');
        assert_eq!(a[p4d(2, -1, 0, 1)], '.');
        assert_eq!(a[p4d(2, 1, 0, 1)], '.');
        assert_eq!(a[p4d(-2, -1, 1, 1)], '.');
        assert_eq!(a[p4d(-2, 1, 1, 1)], '.');
        assert_eq!(a[p4d(2, -1, 1, 1)], '.');
        assert_eq!(a[p4d(2, 1, 1, 1)], '.');
        assert_eq!(a[p4d(-2, -1, 0, 2)], '.');
        assert_eq!(a[p4d(-2, 1, 0, 2)], '.');
        assert_eq!(a[p4d(2, -1, 0, 2)], '.');
        assert_eq!(a[p4d(2, 1, 0, 2)], '.');
        assert_eq!(a[p4d(-2, -1, 1, 2)], '.');
        assert_eq!(a[p4d(-2, 1, 1, 2)], '.');
        assert_eq!(a[p4d(2, -1, 1, 2)], '.');
        assert_eq!(a[p4d(2, 1, 1, 2)], '.');
    }

    #[test]
    fn test_iter_2d() {
        let r = bb2d(0..2, 0..2);
        let a = Array2d::with(r, |p| p.x() + p.y());
        let v = a.iter().cloned().collect::<Vec<_>>();
        assert_eq!(v, vec![0, 1, 1, 2]);
    }
    #[test]
    fn test_iter_3d() {
        let r = bb3d(0..2, 0..2, 0..2);
        let a = Array3d::with(r, |p| p.x() + p.y() + p.z());
        let v = a.iter().cloned().collect::<Vec<_>>();
        assert_eq!(v, vec![0, 1, 1, 2, 1, 2, 2, 3]);
    }
    #[test]
    fn test_iter_4d() {
        let r = bb4d(0..2, 0..2, 0..2, 0..2);
        let a = Array4d::with(r, |p| p.x() + p.y() + p.z() + p.w());
        let v = a.iter().cloned().collect::<Vec<_>>();
        assert_eq!(v, vec![0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]);
    }
}
