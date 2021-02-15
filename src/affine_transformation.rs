//! Affine transformations for n-dimensional discrete spaces.
//!
//! An affine transformation maps points to points
//! so that distances are preserved.  It can be decomposed
//! into a symmetry transformation on the underlying vectors
//! (a rotation or reflection) and a translation.

use crate::Integer;
use crate::Matrix;
use crate::Matrix2d;
use crate::Matrix3d;
use crate::Matrix4d;
use crate::MatrixOps;
use crate::Point;
use crate::Vector;

/// An affine transformation.
///
/// Transforms points to points in an n-dimensional space.
/// Consists of a linear transformation and a translation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AffineTransformation<S, M>
where
    S: Integer,
    M: Matrix<S>,
{
    m: M,
    t: M::V,
}
impl<S, M> AffineTransformation<S, M>
where
    S: Integer,
    M: Matrix<S>,
{
    /// Create a new affine transformation from a matrix and a translation.
    pub fn new(m: M, t: M::V) -> Self {
        AffineTransformation { m, t }
    }

    /// Creates the identity transformation.
    pub fn identity() -> Self {
        Self::new(M::unit(), M::V::zero())
    }

    /// Composes two affine transformations.
    pub fn compose<'a>(&'a self, other: &'a Self) -> Self
    where
        &'a M: MatrixOps<S, &'a M, M>,
        &'a M: MatrixOps<S, M::V, M::V>,
    {
        Self::new(&self.m * &other.m, &other.m * self.t + other.t)
    }

    /// Returns the result of applying the transformation to a point.
    pub fn apply<'a>(&'a self, p: Point<S, M::V>) -> Point<S, M::V>
    where
        &'a M: MatrixOps<S, M::V, M::V>,
    {
        Point::from(&self.m * p.to_vec() + self.t)
    }
}

impl<S, M> Default for AffineTransformation<S, M>
where
    S: Integer,
    M: Matrix<S>,
{
    /// Returns the identity transformation.
    fn default() -> Self {
        Self::identity()
    }
}

/// A 2-dimensional affine transformation.
pub type AffineTransformation2d<S> = AffineTransformation<S, Matrix2d<S>>;

/// A 3-dimensional affine transformation.
pub type AffineTransformation3d<S> = AffineTransformation<S, Matrix3d<S>>;

/// A 4-dimensional affine transformation.
pub type AffineTransformation4d<S> = AffineTransformation<S, Matrix4d<S>>;

#[cfg(test)]
mod tests {
    use crate::p2d;
    use crate::p3d;
    use crate::p4d;
    use crate::v2d;
    use crate::AffineTransformation2d as AT2d;
    use crate::AffineTransformation3d as AT3d;
    use crate::AffineTransformation4d as AT4d;
    use crate::Matrix2d;

    #[test]
    fn test_compose() {
        let at0 = AT2d::new(Matrix2d::new(0, -1, 1, 0), v2d(2, 3));
        let at1 = AT2d::new(Matrix2d::new(0, -1, 1, 0), v2d(4, 5));
        assert_eq!(
            AT2d::new(Matrix2d::new(-1, 0, 0, -1), v2d(1, 7)),
            at0.compose(&at1)
        );
    }

    #[test]
    fn test_apply() {
        let at = AT2d::new(Matrix2d::new(0, -1, 1, 0), v2d(2, 3));
        assert_eq!(p2d(-2, 6), at.apply(p2d(3, 4)));
    }

    #[test]
    fn test_compose_apply_2d() {
        let at0 = AT2d::new(Matrix2d::new(0, -1, 1, 0), v2d(2, 3));
        let at1 = AT2d::new(Matrix2d::new(0, -1, 1, 0), v2d(4, 5));
        let at = at0.compose(&at1);
        assert_eq!(at1.apply(at0.apply(p2d(1, 0))), at.apply(p2d(1, 0)));
        assert_eq!(at1.apply(at0.apply(p2d(0, 1))), at.apply(p2d(0, 1)));
    }

    #[test]
    fn test_identity_2d() {
        let at = AT2d::identity();
        assert_eq!(p2d(1, 0), at.apply(p2d(1, 0)));
        assert_eq!(p2d(0, 1), at.apply(p2d(0, 1)));
    }
    #[test]
    fn test_identity_3d() {
        let at = AT3d::identity();
        assert_eq!(p3d(1, 0, 0), at.apply(p3d(1, 0, 0)));
        assert_eq!(p3d(0, 1, 0), at.apply(p3d(0, 1, 0)));
        assert_eq!(p3d(0, 0, 1), at.apply(p3d(0, 0, 1)));
    }
    #[test]
    fn test_identity_4d() {
        let at = AT4d::identity();
        assert_eq!(p4d(1, 0, 0, 0), at.apply(p4d(1, 0, 0, 0)));
        assert_eq!(p4d(0, 1, 0, 0), at.apply(p4d(0, 1, 0, 0)));
        assert_eq!(p4d(0, 0, 1, 0), at.apply(p4d(0, 0, 1, 0)));
        assert_eq!(p4d(0, 0, 0, 1), at.apply(p4d(0, 0, 0, 1)));
    }
}
