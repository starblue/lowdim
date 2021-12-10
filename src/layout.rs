//! Layout of a multidimensional array in one-dimensional storage.

use std::fmt;

use crate::BBox;
use crate::BBox2d;
use crate::BBox3d;
use crate::BBox4d;
use crate::Integer;
use crate::Point;
use crate::Vec2d;
use crate::Vec3d;
use crate::Vec4d;
use crate::Vector;

/// A layout of a multidimensional slice.
///
/// Points for indexing the multidimensional slice must lie within
/// a given bounding box.
/// The layout maps the points inside the bounding box to indices
/// for use within an array or vector.
/// The map is always injective (different points have different indices).
/// The indices may fill a range starting at zero completely for owned data,
/// or there may be gaps if a multidimensional slice only references part
/// of a multidimensional array.
/// The layout also provides an iterator for the points in the bounding box
/// in order of increasing index.
pub trait Layout<S, V>
where
    Self: Sized + Clone,
    S: Integer,
    V: Vector<S>,
{
    fn new(bbox: BBox<S, V>) -> Self
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug;

    /// Returns the array index of the point if it is inside the boounding box,
    /// or `None` otherwise.
    fn index(&self, p: Point<S, V>) -> Option<usize>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        if self.bbox().contains(&p) {
            let v = p - self.bbox().min();
            let s = v.as_slice();
            Some(
                self.coeffs()
                    .iter()
                    .zip(s.iter())
                    .fold(0_usize, |a, (c, b)| a + c * b.to_usize()),
            )
        } else {
            None
        }
    }

    /// Returns the bounding box of the points mapped by this layout.
    fn bbox(&self) -> BBox<S, V>;

    /// Returns the number of points in this layout.
    fn len(&self) -> usize
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        self.bbox().volume()
    }

    /// Returns an iterator over the points in the bounding box in index order.
    fn into_points(self) -> Points<S, V, Self>;

    /// Returns an iterator over the points in the bounding box
    /// in the order of their internal indices.
    fn points(&self) -> Points<S, V, Self> {
        self.clone().into_points()
    }

    /// Returns the coefficients for index calculation.
    fn coeffs(&self) -> &[usize];
}

#[derive(Clone, Debug)]
pub struct Layout2d<S>
where
    S: Integer,
{
    bbox: BBox2d<S>,
    coeffs: [usize; 2],
}
impl<S> Layout<S, Vec2d<S>> for Layout2d<S>
where
    S: Integer,
{
    /// Returns the default layout for storage without holes starting at index 0.
    ///
    /// Storage is by rows, with the x-coordinate increasing fastest.
    fn new(bbox: BBox2d<S>) -> Layout2d<S>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        let mut coeffs = [0; 2];
        let mut state = 1;
        for (c, len) in coeffs.iter_mut().zip(bbox.lengths().as_slice().iter()) {
            *c = state;
            state *= len.to_usize();
        }
        Layout2d { bbox, coeffs }
    }

    /// Returns the bounding box of the layout.
    fn bbox(&self) -> BBox2d<S> {
        self.bbox
    }

    /// Returns an iterator over the points in the bounding box
    /// in the order of their internal indices.
    fn into_points(self) -> Points<S, Vec2d<S>, Layout2d<S>> {
        Points::new(self)
    }

    /// Returns the coefficients for index calculation.
    fn coeffs(&self) -> &[usize] {
        &self.coeffs
    }
}

#[derive(Clone, Debug)]
pub struct Layout3d<S>
where
    S: Integer,
{
    bbox: BBox3d<S>,
    coeffs: [usize; 3],
}
impl<S> Layout<S, Vec3d<S>> for Layout3d<S>
where
    S: Integer,
{
    /// Returns the default layout for storage without holes starting at index 0.
    ///
    /// Storage is by rows, with the x-coordinate increasing fastest.
    fn new(bbox: BBox3d<S>) -> Layout3d<S>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        let mut coeffs = [0; 3];
        let mut state = 1;
        for (c, len) in coeffs.iter_mut().zip(bbox.lengths().as_slice().iter()) {
            *c = state;
            state *= len.to_usize();
        }
        Layout3d { bbox, coeffs }
    }

    /// Returns the bounding box of the layout.
    fn bbox(&self) -> BBox3d<S> {
        self.bbox
    }

    /// Returns an iterator over the points in the bounding box
    /// in the order of their internal indices.
    fn into_points(self) -> Points<S, Vec3d<S>, Layout3d<S>> {
        Points::new(self)
    }

    /// Returns the coefficients for index calculation.
    fn coeffs(&self) -> &[usize] {
        &self.coeffs
    }
}

#[derive(Clone, Debug)]
pub struct Layout4d<S>
where
    S: Integer,
{
    bbox: BBox4d<S>,
    coeffs: [usize; 4],
}
impl<S> Layout<S, Vec4d<S>> for Layout4d<S>
where
    S: Integer,
{
    /// Returns the default layout for storage without holes starting at index 0.
    ///
    /// Storage is by rows, with the x-coordinate increasing fastest.
    fn new(bbox: BBox4d<S>) -> Layout4d<S>
    where
        usize: TryFrom<S>,
        <usize as TryFrom<S>>::Error: fmt::Debug,
    {
        let mut coeffs = [0; 4];
        let mut state = 1;
        for (c, len) in coeffs.iter_mut().zip(bbox.lengths().as_slice().iter()) {
            *c = state;
            state *= len.to_usize();
        }
        Layout4d { bbox, coeffs }
    }

    /// Returns the bounding box of the layout.
    fn bbox(&self) -> BBox4d<S> {
        self.bbox
    }

    /// Returns an iterator over the points in the bounding box
    /// in the order of their internal indices.
    fn into_points(self) -> Points<S, Vec4d<S>, Layout4d<S>> {
        Points::new(self)
    }

    /// Returns the coefficients for index calculation.
    fn coeffs(&self) -> &[usize] {
        &self.coeffs
    }
}

#[derive(Clone, Debug)]
pub struct Points<S, V, L>
where
    S: Integer,
    V: Vector<S>,
    L: Layout<S, V>,
{
    layout: L,
    next_point: Option<Point<S, V>>,
}
impl<S, V, L> Points<S, V, L>
where
    S: Integer,
    V: Vector<S>,
    L: Layout<S, V>,
{
    fn new(layout: L) -> Points<S, V, L> {
        let next_point = Some(layout.bbox().min());
        Points { layout, next_point }
    }
}
impl<S, V, L> Iterator for Points<S, V, L>
where
    S: Integer,
    V: Vector<S>,
    L: Layout<S, V>,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    type Item = Point<S, V>;

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        let result = self.next_point;
        if let Some(p) = self.next_point {
            let bbox = self.layout.bbox();

            // Move to next point
            let mut p = p;
            let p_coords = p.as_mut_slice();

            let min = bbox.min();
            let min_coords = min.as_slice();

            let max = bbox.max();
            let max_coords = max.as_slice();

            for i in 0..V::DIM {
                if p_coords[i] < max_coords[i] {
                    // Increase this coordinate and be done for this iteration.
                    p_coords[i] += S::one();
                    self.next_point = Some(p);
                    return result;
                } else {
                    // Set this coordinate to minimum and carry on.
                    p_coords[i] = min_coords[i];
                }
            }
            // We found no coordinate to increase, end iterating.
            self.next_point = None;
        }
        result
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = {
            match self.next_point {
                None => 0,
                Some(p) => self.layout.len() - self.layout.index(p).unwrap(),
            }
        };
        (len, Some(len))
    }
}

impl<S, V, L> ExactSizeIterator for Points<S, V, L>
where
    S: Integer,
    V: Vector<S>,
    L: Layout<S, V>,
    usize: TryFrom<S>,
    <usize as TryFrom<S>>::Error: fmt::Debug,
{
    fn len(&self) -> usize {
        self.size_hint().0
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
    use crate::Layout;
    use crate::Layout2d;
    use crate::Layout3d;
    use crate::Layout4d;

    #[test]
    fn test_index_2d() {
        let bbox = bb2d(0..2, -2..0);
        let layout = Layout2d::new(bbox);
        assert_eq!(Some(0), layout.index(p2d(0, -2)));
        assert_eq!(Some(1), layout.index(p2d(1, -2)));
        assert_eq!(Some(2), layout.index(p2d(0, -1)));
        assert_eq!(Some(3), layout.index(p2d(1, -1)));
        assert_eq!(None, layout.index(p2d(-1, -1)));
        assert_eq!(None, layout.index(p2d(2, -1)));
        assert_eq!(None, layout.index(p2d(0, -3)));
        assert_eq!(None, layout.index(p2d(0, 0)));
    }
    #[test]
    fn test_index_3d() {
        let bbox = bb3d(0..2, -2..0, 1..3);
        let layout = Layout3d::new(bbox);
        assert_eq!(Some(0), layout.index(p3d(0, -2, 1)));
        assert_eq!(Some(1), layout.index(p3d(1, -2, 1)));
        assert_eq!(Some(2), layout.index(p3d(0, -1, 1)));
        assert_eq!(Some(3), layout.index(p3d(1, -1, 1)));
        assert_eq!(Some(4), layout.index(p3d(0, -2, 2)));
        assert_eq!(Some(5), layout.index(p3d(1, -2, 2)));
        assert_eq!(Some(6), layout.index(p3d(0, -1, 2)));
        assert_eq!(Some(7), layout.index(p3d(1, -1, 2)));
        assert_eq!(None, layout.index(p3d(-1, -1, 1)));
        assert_eq!(None, layout.index(p3d(2, -1, 1)));
        assert_eq!(None, layout.index(p3d(0, -3, 1)));
        assert_eq!(None, layout.index(p3d(0, 0, 1)));
        assert_eq!(None, layout.index(p3d(0, -1, 0)));
        assert_eq!(None, layout.index(p3d(0, -1, 3)));
    }
    #[test]
    fn test_index_4d() {
        let bbox = bb4d(0..2, -2..0, 1..3, -5..-3);
        let layout = Layout4d::new(bbox);
        assert_eq!(Some(0), layout.index(p4d(0, -2, 1, -5)));
        assert_eq!(Some(1), layout.index(p4d(1, -2, 1, -5)));
        assert_eq!(Some(2), layout.index(p4d(0, -1, 1, -5)));
        assert_eq!(Some(3), layout.index(p4d(1, -1, 1, -5)));
        assert_eq!(Some(4), layout.index(p4d(0, -2, 2, -5)));
        assert_eq!(Some(5), layout.index(p4d(1, -2, 2, -5)));
        assert_eq!(Some(6), layout.index(p4d(0, -1, 2, -5)));
        assert_eq!(Some(7), layout.index(p4d(1, -1, 2, -5)));
        assert_eq!(Some(8), layout.index(p4d(0, -2, 1, -4)));
        assert_eq!(Some(9), layout.index(p4d(1, -2, 1, -4)));
        assert_eq!(Some(10), layout.index(p4d(0, -1, 1, -4)));
        assert_eq!(Some(11), layout.index(p4d(1, -1, 1, -4)));
        assert_eq!(Some(12), layout.index(p4d(0, -2, 2, -4)));
        assert_eq!(Some(13), layout.index(p4d(1, -2, 2, -4)));
        assert_eq!(Some(14), layout.index(p4d(0, -1, 2, -4)));
        assert_eq!(Some(15), layout.index(p4d(1, -1, 2, -4)));
        assert_eq!(None, layout.index(p4d(-1, -1, 1, -5)));
        assert_eq!(None, layout.index(p4d(2, -1, 1, -5)));
        assert_eq!(None, layout.index(p4d(0, -3, 1, -5)));
        assert_eq!(None, layout.index(p4d(0, 0, 1, -5)));
        assert_eq!(None, layout.index(p4d(0, -1, 0, -5)));
        assert_eq!(None, layout.index(p4d(0, -1, 3, -5)));
        assert_eq!(None, layout.index(p4d(0, -1, 1, -6)));
        assert_eq!(None, layout.index(p4d(0, -1, 1, -3)));
    }

    #[test]
    fn test_points_2d() {
        let bbox = bb2d(0..2, -2..0);
        let layout = Layout2d::new(bbox);
        let mut count = 0;
        for (i, p) in layout.points().enumerate() {
            assert_eq!(Some(i), layout.index(p));
            count += 1;
        }
        assert_eq!(layout.bbox.volume(), count);
    }
    #[test]
    fn test_points_3d() {
        let bbox = bb3d(0..2, -2..0, 1..3);
        let layout = Layout3d::new(bbox);
        let mut count = 0;
        for (i, p) in layout.points().enumerate() {
            assert_eq!(Some(i), layout.index(p));
            count += 1;
        }
        assert_eq!(layout.bbox.volume(), count);
    }
    #[test]
    fn test_points_4d() {
        let bbox = bb4d(0..2, -2..0, 1..3, -5..-3);
        let layout = Layout4d::new(bbox);
        let mut count = 0;
        for (i, p) in layout.points().enumerate() {
            assert_eq!(Some(i), layout.index(p));
            count += 1;
        }
        assert_eq!(layout.bbox.volume(), count);
    }
}
