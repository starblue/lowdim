use core::ops::Add;
use core::ops::AddAssign;
use core::ops::Index;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Sub;
use core::ops::SubAssign;

use crate::Integer;
use crate::Vector;
use crate::VectorOps;

const DIM: usize = 3;

/// A three-dimensional discrete vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Vec3d<S = i64>([S; DIM])
where
    S: Integer;

impl<S: Integer> Vec3d<S> {
    /// Creates a new 3d-vector from its coordinates.
    pub fn new(x: S, y: S, z: S) -> Vec3d<S> {
        Vec3d([x, y, z])
    }

    pub fn x(&self) -> S {
        self.0[0]
    }
    pub fn y(&self) -> S {
        self.0[1]
    }
    pub fn z(&self) -> S {
        self.0[2]
    }
}

impl<S: Integer> VectorOps<S, Vec3d<S>> for Vec3d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec3d<S>> for Vec3d<S> {}

impl<S: Integer> Vector<S> for Vec3d<S> {
    fn with<F>(f: F) -> Vec3d<S>
    where
        F: Fn(usize) -> S,
    {
        Vec3d([f(0), f(1), f(2)])
    }

    /// The L1, taxicab or Manhatten norm.
    fn norm_l1(&self) -> S {
        self.x().abs() + self.y().abs() + self.z().abs()
    }
    /// Creates a vector of the 6 orthogonal vectors, i.e. those with L1 norm equal to 1.
    fn unit_vecs_l1() -> Vec<Self> {
        vec![
            v3d(1, 0, 0),
            v3d(0, 1, 0),
            v3d(0, 0, 1),
            v3d(-1, 0, 0),
            v3d(0, -1, 0),
            v3d(0, 0, -1),
        ]
    }

    /// The maximum, Chebychev or L-infinity norm.
    fn norm_infty(&self) -> S {
        Integer::max(Integer::max(self.x().abs(), self.y().abs()), self.z().abs())
    }
    /// Creates a vector of the 26 vectors with L∞ norm equal to 1.
    fn unit_vecs_l_infty() -> Vec<Self> {
        let mut result = Vec::new();
        for z in -1..=1 {
            for y in -1..=1 {
                for x in -1..=1 {
                    if x != 0 || y != 0 || z != 0 {
                        result.push(v3d(x, y, z));
                    }
                }
            }
        }
        result
    }
}

/// Creates a 3d-vector.
///
/// This is a utility function for concisely representing vectors.
pub fn v3d<S: Integer, T>(x: T, y: T, z: T) -> Vec3d<S>
where
    S: From<T>,
{
    Vec3d::new(S::from(x), S::from(y), S::from(z))
}

impl<S: Integer> Index<usize> for Vec3d<S> {
    type Output = S;
    fn index(&self, i: usize) -> &S {
        self.0.index(i)
    }
}

impl<S: Integer> Add<Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> Sub<Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        Vec3d::with(|i| self[i] - other[i])
    }
}

impl<S: Integer> Neg for Vec3d<S> {
    type Output = Vec3d<S>;

    fn neg(self) -> Vec3d<S> {
        Vec3d::with(|i| -self[i])
    }
}
impl<'a, S: Integer> Neg for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn neg(self) -> Vec3d<S> {
        Vec3d::with(|i| -self[i])
    }
}

impl Mul<Vec3d<i64>> for i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: Vec3d<i64>) -> Vec3d<i64> {
        Vec3d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec3d<i64>> for i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: &'a Vec3d<i64>) -> Vec3d<i64> {
        Vec3d::with(|i| self * other[i])
    }
}

impl<'a> Mul<Vec3d<i64>> for &'a i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: Vec3d<i64>) -> Vec3d<i64> {
        Vec3d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec3d<i64>> for &'a i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: &'a Vec3d<i64>) -> Vec3d<i64> {
        Vec3d::with(|i| self * other[i])
    }
}

impl<S: Integer> Mul<Vec3d<S>> for Vec3d<S> {
    type Output = S;

    fn mul(self, other: Vec3d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl<'a, S: Integer> Mul<&'a Vec3d<S>> for Vec3d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec3d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl<'a, S: Integer> Mul<Vec3d<S>> for &'a Vec3d<S> {
    type Output = S;

    fn mul(self, other: Vec3d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl<'a, S: Integer> Mul<&'a Vec3d<S>> for &'a Vec3d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec3d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
    }
}

impl<S: Integer> AddAssign for Vec3d<S> {
    fn add_assign(&mut self, other: Vec3d<S>) {
        *self = Vec3d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> AddAssign<&'a Vec3d<S>> for Vec3d<S> {
    fn add_assign(&mut self, other: &'a Vec3d<S>) {
        *self = Vec3d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> SubAssign for Vec3d<S> {
    fn sub_assign(&mut self, other: Vec3d<S>) {
        *self = Vec3d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> SubAssign<&'a Vec3d<S>> for Vec3d<S> {
    fn sub_assign(&mut self, other: &'a Vec3d<S>) {
        *self = Vec3d::with(|i| self[i] - other[i])
    }
}
