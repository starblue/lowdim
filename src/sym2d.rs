//! Symmetries of 2d vectors.

use core::fmt;
use core::ops::Mul;

use crate::matrix::Matrix;
use crate::Integer;
use crate::Matrix2d;

/// The symmetries of discrete 2d vectors.
///
/// These preserve length and the origin, and correspond
/// to multiplication with orthogonal integer matrices.
/// This can be viewed as a concise representation of the
/// orthogonal matrices, conversion is possible via [Matrix2d::from].
///
/// This is the dihedral group D4, the symmetry group of a square.
/// It consists of four rotations and four reflections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sym2d {
    /// Rotates by the given number of right angles to the left.
    R(u8),
    /// Reflects by an axis through the origin so that
    /// the point (1, 0) is rotated by the given number of right angles to the left.
    ///
    /// The axis of reflection then forms an angle of 45 degrees times the given number
    /// with the x axis.
    S(u8),
}
impl Sym2d {
    /// Returns a vector containing the elements of the symmetry group.
    pub fn elements() -> Vec<Sym2d> {
        vec![
            Sym2d::R(0),
            Sym2d::R(1),
            Sym2d::R(2),
            Sym2d::R(3),
            Sym2d::S(0),
            Sym2d::S(1),
            Sym2d::S(2),
            Sym2d::S(3),
        ]
    }
    /// Returns a vector containing the rotations of the symmetry group.
    ///
    /// The rotations are closed under multiplication.
    /// That is, they form a subgroup.
    pub fn rotations() -> Vec<Sym2d> {
        vec![Sym2d::R(0), Sym2d::R(1), Sym2d::R(2), Sym2d::R(3)]
    }
    /// Returns the inverse of an element.
    pub fn inv(&self) -> Sym2d {
        match self {
            Sym2d::R(i) => Sym2d::R((4 - i) % 4),
            Sym2d::S(i) => Sym2d::S(*i),
        }
    }
    /// Returns true if this a rotation.
    pub fn is_rotation(&self) -> bool {
        match self {
            Sym2d::R(_) => true,
            Sym2d::S(_) => false,
        }
    }
    /// Returns true if this a reflection.
    pub fn is_reflection(&self) -> bool {
        match self {
            Sym2d::R(_) => false,
            Sym2d::S(_) => true,
        }
    }
}

impl fmt::Display for Sym2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sym2d::R(i) => write!(f, "r{}", i),
            Sym2d::S(i) => write!(f, "s{}", i),
        }
    }
}

impl<S: Integer> From<Sym2d> for Matrix2d<S> {
    fn from(s: Sym2d) -> Self {
        match s {
            Sym2d::R(0) => Matrix2d::unit(),
            Sym2d::R(1) => Matrix2d::rotate_left_90(),
            Sym2d::R(2) => Matrix2d::rotate_180(),
            Sym2d::R(3) => Matrix2d::rotate_right_90(),
            Sym2d::S(0) => Matrix2d::reflect_x_axis(),
            Sym2d::S(1) => Matrix2d::reflect_diagonal(),
            Sym2d::S(2) => Matrix2d::reflect_y_axis(),
            Sym2d::S(3) => Matrix2d::reflect_anti_diagonal(),
            _ => panic!("unexpected 2d symmetry"),
        }
    }
}

impl Mul for Sym2d {
    type Output = Sym2d;
    fn mul(self, other: Sym2d) -> Sym2d {
        match (self, other) {
            (Sym2d::R(r0), Sym2d::R(r1)) => Sym2d::R((r0 + r1) % 4),
            (Sym2d::R(r0), Sym2d::S(r1)) => Sym2d::S((r0 + r1) % 4),
            (Sym2d::S(r0), Sym2d::R(r1)) => Sym2d::S((r0 + 4 - r1) % 4),
            (Sym2d::S(r0), Sym2d::S(r1)) => Sym2d::R((r0 + 4 - r1) % 4),
        }
    }
}
impl<'a> Mul<&'a Sym2d> for Sym2d {
    type Output = Sym2d;
    fn mul(self, other: &Sym2d) -> Sym2d {
        self * *other
    }
}
impl<'a> Mul<Sym2d> for &'a Sym2d {
    type Output = Sym2d;
    fn mul(self, other: Sym2d) -> Sym2d {
        *self * other
    }
}
impl<'a> Mul<&'a Sym2d> for &'a Sym2d {
    type Output = Sym2d;
    fn mul(self, other: &Sym2d) -> Sym2d {
        *self * *other
    }
}

#[cfg(test)]
mod tests {
    use crate::Matrix2d;

    use super::Sym2d;

    #[test]
    fn test_mul() {
        for s0 in Sym2d::elements() {
            for s1 in Sym2d::elements() {
                let m0: Matrix2d<i64> = Matrix2d::from(s0);
                let m1: Matrix2d<i64> = Matrix2d::from(s1);
                assert_eq!(Matrix2d::from(s0 * s1), m0 * m1);
                assert_eq!(Matrix2d::from(s0 * &s1), m0 * m1);
                assert_eq!(Matrix2d::from(&s0 * s1), m0 * m1);
                assert_eq!(Matrix2d::from(&s0 * &s1), m0 * m1);
            }
        }
    }

    #[test]
    fn test_inv() {
        for s in Sym2d::elements() {
            assert_eq!(Sym2d::R(0), s * s.inv());
            assert_eq!(Sym2d::R(0), s.inv() * s);
        }
    }

    #[test]
    fn test_rotations() {
        let es = Sym2d::rotations();
        assert_eq!(Sym2d::R(0), es[0]);
        assert_eq!(Sym2d::R(1), es[1]);
        assert_eq!(Sym2d::R(2), es[2]);
        assert_eq!(Sym2d::R(3), es[3]);
    }

    #[test]
    fn test_is_rotation() {
        assert!(Sym2d::R(1).is_rotation());
        assert!(!Sym2d::S(1).is_rotation());
    }
    #[test]
    fn test_is_reflection() {
        assert!(!Sym2d::R(1).is_reflection());
        assert!(Sym2d::S(1).is_reflection());
    }

    #[test]
    fn test_display() {
        assert_eq!("r0", format!("{}", Sym2d::R(0)));
        assert_eq!("s0", format!("{}", Sym2d::S(0)));
    }
}
