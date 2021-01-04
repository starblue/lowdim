use core::ops::Add;
use core::ops::AddAssign;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Sub;
use core::ops::SubAssign;

use crate::Integer;
use crate::Vector;
use crate::VectorOps;

/// A two-dimensional discrete vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Vec3d<S = i64>
where
    S: Integer,
{
    x: S,
    y: S,
    z: S,
}

impl<S: Integer> Vec3d<S> {
    /// Creates a new 3d-vector from its coordinates.
    pub fn new(x: S, y: S, z: S) -> Vec3d<S> {
        Vec3d { x, y, z }
    }
    pub fn x(&self) -> S {
        self.x
    }
    pub fn y(&self) -> S {
        self.y
    }
    pub fn z(&self) -> S {
        self.z
    }
}

impl<S: Integer> VectorOps<S, Vec3d<S>> for Vec3d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec3d<S>> for Vec3d<S> {}

impl<S: Integer> Vector<S> for Vec3d<S> {
    fn min(&self, other: Self) -> Self {
        Vec3d {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }
    fn max(&self, other: Self) -> Self {
        Vec3d {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }
    /// The L1, taxicab or Manhatten norm.
    fn norm_l1(&self) -> S {
        self.x.abs() + self.y.abs() + self.z.abs()
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
    fn norm_infty(&self) -> S
    where
        S: Ord,
    {
        self.x.max(self.y).max(self.z)
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

impl<S: Integer> Add<Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: Vec3d<S>) -> Vec3d<S> {
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = self.z + other.z;
        Vec3d { x, y, z }
    }
}

impl<'a, S: Integer> Add<&'a Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = self.z + other.z;
        Vec3d { x, y, z }
    }
}

impl<'a, S: Integer> Add<Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: Vec3d<S>) -> Vec3d<S> {
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = self.z + other.z;
        Vec3d { x, y, z }
    }
}

impl<'a, S: Integer> Add<&'a Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn add(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = self.z + other.z;
        Vec3d { x, y, z }
    }
}

impl<S: Integer> Sub<Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: Vec3d<S>) -> Vec3d<S> {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let z = self.z - other.z;
        Vec3d { x, y, z }
    }
}

impl<'a, S: Integer> Sub<&'a Vec3d<S>> for Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let z = self.z - other.z;
        Vec3d { x, y, z }
    }
}

impl<'a, S: Integer> Sub<Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: Vec3d<S>) -> Vec3d<S> {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let z = self.z - other.z;
        Vec3d { x, y, z }
    }
}

impl<'a, S: Integer> Sub<&'a Vec3d<S>> for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn sub(self, other: &'a Vec3d<S>) -> Vec3d<S> {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let z = self.z - other.z;
        Vec3d { x, y, z }
    }
}

impl<S: Integer> Neg for Vec3d<S> {
    type Output = Vec3d<S>;

    fn neg(self) -> Vec3d<S> {
        let x = -self.x;
        let y = -self.y;
        let z = -self.z;
        Vec3d { x, y, z }
    }
}
impl<'a, S: Integer> Neg for &'a Vec3d<S> {
    type Output = Vec3d<S>;

    fn neg(self) -> Vec3d<S> {
        let x = -self.x;
        let y = -self.y;
        let z = -self.z;
        Vec3d { x, y, z }
    }
}

impl Mul<Vec3d<i64>> for i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: Vec3d<i64>) -> Vec3d<i64> {
        let x = self * other.x;
        let y = self * other.y;
        let z = self * other.z;
        Vec3d { x, y, z }
    }
}

impl<'a> Mul<&'a Vec3d<i64>> for i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: &'a Vec3d<i64>) -> Vec3d<i64> {
        let x = self * other.x;
        let y = self * other.y;
        let z = self * other.z;
        Vec3d { x, y, z }
    }
}

impl<'a> Mul<Vec3d<i64>> for &'a i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: Vec3d<i64>) -> Vec3d<i64> {
        let x = self * other.x;
        let y = self * other.y;
        let z = self * other.z;
        Vec3d { x, y, z }
    }
}

impl<'a> Mul<&'a Vec3d<i64>> for &'a i64 {
    type Output = Vec3d<i64>;

    fn mul(self, other: &'a Vec3d<i64>) -> Vec3d<i64> {
        let x = self * other.x;
        let y = self * other.y;
        let z = self * other.z;
        Vec3d { x, y, z }
    }
}

impl<S: Integer> AddAssign for Vec3d<S> {
    fn add_assign(&mut self, other: Vec3d<S>) {
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = self.z + other.z;
        *self = Vec3d { x, y, z }
    }
}

impl<'a, S: Integer> AddAssign<&'a Vec3d<S>> for Vec3d<S> {
    fn add_assign(&mut self, other: &'a Vec3d<S>) {
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = self.z + other.z;
        *self = Vec3d { x, y, z }
    }
}

impl<S: Integer> SubAssign for Vec3d<S> {
    fn sub_assign(&mut self, other: Vec3d<S>) {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let z = self.z - other.z;
        *self = Vec3d { x, y, z }
    }
}

impl<'a, S: Integer> SubAssign<&'a Vec3d<S>> for Vec3d<S> {
    fn sub_assign(&mut self, other: &'a Vec3d<S>) {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let z = self.z - other.z;
        *self = Vec3d { x, y, z }
    }
}
