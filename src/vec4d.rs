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

const DIM: usize = 4;

/// A four-dimensional discrete vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Vec4d<S = i64>([S; DIM])
where
    S: Integer;

impl<S: Integer> Vec4d<S> {
    /// Creates a new 4d-vector from its coordinates.
    pub fn new(x: S, y: S, z: S, w: S) -> Vec4d<S> {
        Vec4d([x, y, z, w])
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
    pub fn w(&self) -> S {
        self.0[3]
    }
}

impl<S: Integer> VectorOps<S, Vec4d<S>> for Vec4d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec4d<S>> for Vec4d<S> {}

impl<S: Integer> Vector<S> for Vec4d<S> {
    fn with<F>(f: F) -> Vec4d<S>
    where
        F: Fn(usize) -> S,
    {
        Vec4d([f(0), f(1), f(2), f(3)])
    }

    /// The L1, taxicab or Manhatten norm.
    fn norm_l1(&self) -> S {
        self.x().abs() + self.y().abs() + self.z().abs() + self.w().abs()
    }
    /// Creates a vector of the 6 orthogonal vectors, i.e. those with L1 norm equal to 1.
    fn unit_vecs_l1() -> Vec<Self> {
        vec![
            v4d(1, 0, 0, 0),
            v4d(0, 1, 0, 0),
            v4d(0, 0, 1, 0),
            v4d(0, 0, 0, 1),
            v4d(-1, 0, 0, 0),
            v4d(0, -1, 0, 0),
            v4d(0, 0, -1, 0),
            v4d(0, 0, 0, -1),
        ]
    }

    /// The maximum, Chebychev or L-infinity norm.
    fn norm_infty(&self) -> S {
        Integer::max(
            Integer::max(Integer::max(self.x().abs(), self.y().abs()), self.z().abs()),
            self.w().abs(),
        )
    }
    /// Creates a vector of the 26 vectors with L∞ norm equal to 1.
    fn unit_vecs_l_infty() -> Vec<Self> {
        let mut result = Vec::new();
        for w in -1..=1 {
            for z in -1..=1 {
                for y in -1..=1 {
                    for x in -1..=1 {
                        if x != 0 || y != 0 || z != 0 || w != 0 {
                            result.push(v4d(x, y, z, w));
                        }
                    }
                }
            }
        }
        result
    }
}

/// Creates a 4d-vector.
///
/// This is a utility function for concisely representing vectors.
pub fn v4d<S: Integer, T>(x: T, y: T, z: T, w: T) -> Vec4d<S>
where
    S: From<T>,
{
    Vec4d::new(S::from(x), S::from(y), S::from(z), S::from(w))
}

impl<S: Integer> Index<usize> for Vec4d<S> {
    type Output = S;
    fn index(&self, i: usize) -> &S {
        self.0.index(i)
    }
}

impl<S: Integer> Add<Vec4d<S>> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn add(self, other: Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec4d<S>> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn add(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<Vec4d<S>> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn add(self, other: Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> Add<&'a Vec4d<S>> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn add(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> Sub<Vec4d<S>> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn sub(self, other: Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec4d<S>> for Vec4d<S> {
    type Output = Vec4d<S>;

    fn sub(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<Vec4d<S>> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn sub(self, other: Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> Sub<&'a Vec4d<S>> for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn sub(self, other: &'a Vec4d<S>) -> Vec4d<S> {
        Vec4d::with(|i| self[i] - other[i])
    }
}

impl<S: Integer> Neg for Vec4d<S> {
    type Output = Vec4d<S>;

    fn neg(self) -> Vec4d<S> {
        Vec4d::with(|i| -self[i])
    }
}
impl<'a, S: Integer> Neg for &'a Vec4d<S> {
    type Output = Vec4d<S>;

    fn neg(self) -> Vec4d<S> {
        Vec4d::with(|i| -self[i])
    }
}
impl Mul<Vec4d<i64>> for i64 {
    type Output = Vec4d<i64>;

    fn mul(self, other: Vec4d<i64>) -> Vec4d<i64> {
        Vec4d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec4d<i64>> for i64 {
    type Output = Vec4d<i64>;

    fn mul(self, other: &'a Vec4d<i64>) -> Vec4d<i64> {
        Vec4d::with(|i| self * other[i])
    }
}

impl<'a> Mul<Vec4d<i64>> for &'a i64 {
    type Output = Vec4d<i64>;

    fn mul(self, other: Vec4d<i64>) -> Vec4d<i64> {
        Vec4d::with(|i| self * other[i])
    }
}

impl<'a> Mul<&'a Vec4d<i64>> for &'a i64 {
    type Output = Vec4d<i64>;

    fn mul(self, other: &'a Vec4d<i64>) -> Vec4d<i64> {
        Vec4d::with(|i| self * other[i])
    }
}

impl<S: Integer> Mul<Vec4d<S>> for Vec4d<S> {
    type Output = S;

    fn mul(self, other: Vec4d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z() + self.w() * other.w()
    }
}

impl<'a, S: Integer> Mul<&'a Vec4d<S>> for Vec4d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec4d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z() + self.w() * other.w()
    }
}

impl<'a, S: Integer> Mul<Vec4d<S>> for &'a Vec4d<S> {
    type Output = S;

    fn mul(self, other: Vec4d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z() + self.w() * other.w()
    }
}

impl<'a, S: Integer> Mul<&'a Vec4d<S>> for &'a Vec4d<S> {
    type Output = S;

    fn mul(self, other: &'a Vec4d<S>) -> S {
        self.x() * other.x() + self.y() * other.y() + self.z() * other.z() + self.w() * other.w()
    }
}

impl<S: Integer> AddAssign for Vec4d<S> {
    fn add_assign(&mut self, other: Vec4d<S>) {
        *self = Vec4d::with(|i| self[i] + other[i])
    }
}

impl<'a, S: Integer> AddAssign<&'a Vec4d<S>> for Vec4d<S> {
    fn add_assign(&mut self, other: &'a Vec4d<S>) {
        *self = Vec4d::with(|i| self[i] + other[i])
    }
}

impl<S: Integer> SubAssign for Vec4d<S> {
    fn sub_assign(&mut self, other: Vec4d<S>) {
        *self = Vec4d::with(|i| self[i] - other[i])
    }
}

impl<'a, S: Integer> SubAssign<&'a Vec4d<S>> for Vec4d<S> {
    fn sub_assign(&mut self, other: &'a Vec4d<S>) {
        *self = Vec4d::with(|i| self[i] - other[i])
    }
}
