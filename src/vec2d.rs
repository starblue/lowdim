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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Vec2d<S = i64>
where
    S: Integer,
{
    x: S,
    y: S,
}

impl<S: Integer> Vec2d<S> {
    /// Creates a new 2d-vector from its coordinates.
    pub fn new(x: S, y: S) -> Vec2d<S> {
        Vec2d { x, y }
    }
    pub fn x(&self) -> S {
        self.x
    }
    pub fn y(&self) -> S {
        self.y
    }
    pub fn rotate_left(&self) -> Self {
        Vec2d {
            x: -self.y,
            y: self.x,
        }
    }
    pub fn rotate_right(&self) -> Self {
        Vec2d {
            x: self.y,
            y: -self.x,
        }
    }
}

impl<S: Integer> VectorOps<S, Vec2d<S>> for Vec2d<S> {}
impl<'a, S: Integer> VectorOps<S, &'a Vec2d<S>> for Vec2d<S> {}

impl<S: Integer> Vector<S> for Vec2d<S> {
    fn min(&self, other: Self) -> Self {
        Vec2d {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }
    fn max(&self, other: Self) -> Self {
        Vec2d {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }
    /// The L1, taxicab or Manhatten norm.
    fn norm_l1(&self) -> S {
        self.x.abs() + self.y.abs()
    }
    /// Creates a vector of the 4 orthogonal vectors, i.e. those with L1 norm equal to 1.
    fn unit_vecs_l1() -> Vec<Self> {
        vec![v2d(1, 0), v2d(0, 1), v2d(-1, 0), v2d(0, -1)]
    }

    /// The maximum, Chebychev or L∞ norm.
    fn norm_infty(&self) -> S {
        self.x.max(self.y)
    }
    /// Creates a vector of the 8 vectors with L∞ norm equal to 1.
    fn unit_vecs_l_infty() -> Vec<Self> {
        let mut result = Vec::new();
        for y in -1..=1 {
            for x in -1..=1 {
                if x != 0 || y != 0 {
                    result.push(v2d(x, y));
                }
            }
        }
        result
    }
}

/// Creates a 2d-vector.
///
/// This is a utility function for concisely representing vectors.
pub fn v2d<S: Integer, T>(x: T, y: T) -> Vec2d<S>
where
    S: From<T>,
{
    Vec2d::new(S::from(x), S::from(y))
}

impl<S: Integer> Add<Vec2d<S>> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn add(self, other: Vec2d<S>) -> Vec2d<S> {
        let x = self.x + other.x;
        let y = self.y + other.y;
        Vec2d { x, y }
    }
}

impl<'a, S: Integer> Add<&'a Vec2d<S>> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn add(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        let x = self.x + other.x;
        let y = self.y + other.y;
        Vec2d { x, y }
    }
}

impl<'a, S: Integer> Add<Vec2d<S>> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn add(self, other: Vec2d<S>) -> Vec2d<S> {
        let x = self.x + other.x;
        let y = self.y + other.y;
        Vec2d { x, y }
    }
}

impl<'a, S: Integer> Add<&'a Vec2d<S>> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn add(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        let x = self.x + other.x;
        let y = self.y + other.y;
        Vec2d { x, y }
    }
}

impl<S: Integer> Sub<Vec2d<S>> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn sub(self, other: Vec2d<S>) -> Vec2d<S> {
        let x = self.x - other.x;
        let y = self.y - other.y;
        Vec2d { x, y }
    }
}

impl<'a, S: Integer> Sub<&'a Vec2d<S>> for Vec2d<S> {
    type Output = Vec2d<S>;

    fn sub(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        let x = self.x - other.x;
        let y = self.y - other.y;
        Vec2d { x, y }
    }
}

impl<'a, S: Integer> Sub<Vec2d<S>> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn sub(self, other: Vec2d<S>) -> Vec2d<S> {
        let x = self.x - other.x;
        let y = self.y - other.y;
        Vec2d { x, y }
    }
}

impl<'a, S: Integer> Sub<&'a Vec2d<S>> for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn sub(self, other: &'a Vec2d<S>) -> Vec2d<S> {
        let x = self.x - other.x;
        let y = self.y - other.y;
        Vec2d { x, y }
    }
}

impl<S: Integer> Neg for Vec2d<S> {
    type Output = Vec2d<S>;

    fn neg(self) -> Vec2d<S> {
        let x = -self.x;
        let y = -self.y;
        Vec2d { x, y }
    }
}
impl<'a, S: Integer> Neg for &'a Vec2d<S> {
    type Output = Vec2d<S>;

    fn neg(self) -> Vec2d<S> {
        let x = -self.x;
        let y = -self.y;
        Vec2d { x, y }
    }
}

impl Mul<Vec2d<i64>> for i64 {
    type Output = Vec2d<i64>;

    fn mul(self, other: Vec2d<i64>) -> Vec2d<i64> {
        let x = self * other.x;
        let y = self * other.y;
        Vec2d { x, y }
    }
}

impl<'a> Mul<&'a Vec2d<i64>> for i64 {
    type Output = Vec2d<i64>;

    fn mul(self, other: &'a Vec2d<i64>) -> Vec2d<i64> {
        let x = self * other.x;
        let y = self * other.y;
        Vec2d { x, y }
    }
}

impl<'a> Mul<Vec2d<i64>> for &'a i64 {
    type Output = Vec2d<i64>;

    fn mul(self, other: Vec2d<i64>) -> Vec2d<i64> {
        let x = self * other.x;
        let y = self * other.y;
        Vec2d { x, y }
    }
}

impl<'a> Mul<&'a Vec2d<i64>> for &'a i64 {
    type Output = Vec2d<i64>;

    fn mul(self, other: &'a Vec2d<i64>) -> Vec2d<i64> {
        let x = self * other.x;
        let y = self * other.y;
        Vec2d { x, y }
    }
}

impl<S: Integer> AddAssign for Vec2d<S> {
    fn add_assign(&mut self, other: Vec2d<S>) {
        let x = self.x + other.x;
        let y = self.y + other.y;
        *self = Vec2d { x, y }
    }
}

impl<'a, S: Integer> AddAssign<&'a Vec2d<S>> for Vec2d<S> {
    fn add_assign(&mut self, other: &'a Vec2d<S>) {
        let x = self.x + other.x;
        let y = self.y + other.y;
        *self = Vec2d { x, y }
    }
}

impl<S: Integer> SubAssign for Vec2d<S> {
    fn sub_assign(&mut self, other: Vec2d<S>) {
        let x = self.x - other.x;
        let y = self.y - other.y;
        *self = Vec2d { x, y }
    }
}

impl<'a, S: Integer> SubAssign<&'a Vec2d<S>> for Vec2d<S> {
    fn sub_assign(&mut self, other: &'a Vec2d<S>) {
        let x = self.x - other.x;
        let y = self.y - other.y;
        *self = Vec2d { x, y }
    }
}
