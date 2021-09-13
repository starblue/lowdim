//! Affine geometry for discrete 2d, 3d and 4d worlds.
//!
//! Provides vectors, matrices, points and more to simplify
//! programming for 2d to 4d worlds.
//! Features include:
//! * Bounding boxes and iterators over the points inside them
//! * Arrays indexed by points
//! * Rotations and reflections
//! * Taxicab (L1) and maximum (L∞) metrics
//! * Orthogonal and king's move neighbourhoods
//!
//! This was originally motivated by
//! [Advent of Code](https://adventofcode.com/)
//! (yes, Santa's elves use the 4th dimension, for example for
//! [time travel](https://adventofcode.com/2018/day/25)).
//!
//! ## Crate Status
//!
//! Experimental
//!
//! ### Limitations
//! - Bounding boxes and arrays are only implemented for the 2d case.
//! - Rotations are only implemented for the 2d case, reflections not at all.
//!
//! ## Crate Feature Flags
//!
//! - `random`
//!   - Off by default.
//!   - Adds a dependency to the `rand` crate.
//!   - Adds the method `BBox2d::random_point` to generate random points
//!     in a bounding box.

#![warn(missing_docs)]

pub mod integer;
#[doc(inline)]
pub use integer::Integer;

pub mod vector;
#[doc(inline)]
pub use vector::Vector;
#[doc(inline)]
pub use vector::VectorOps;

pub mod vec2d;
#[doc(inline)]
pub use crate::vec2d::v2d;
#[doc(inline)]
pub use crate::vec2d::Vec2d;

pub mod vec3d;
#[doc(inline)]
pub use crate::vec3d::v3d;
#[doc(inline)]
pub use crate::vec3d::Vec3d;

pub mod vec4d;
#[doc(inline)]
pub use crate::vec4d::v4d;
#[doc(inline)]
pub use crate::vec4d::Vec4d;

pub mod matrix;
#[doc(inline)]
pub use matrix::Matrix;
#[doc(inline)]
pub use matrix::MatrixOps;

pub mod matrix2d;
#[doc(inline)]
pub use crate::matrix2d::Matrix2d;

pub mod matrix3d;
#[doc(inline)]
pub use crate::matrix3d::Matrix3d;

pub mod matrix4d;
#[doc(inline)]
pub use crate::matrix4d::Matrix4d;

pub mod point;
#[doc(inline)]
pub use crate::point::p2d;
#[doc(inline)]
pub use crate::point::p3d;
#[doc(inline)]
pub use crate::point::p4d;
#[doc(inline)]
pub use crate::point::Point;
#[doc(inline)]
pub use crate::point::Point2d;
#[doc(inline)]
pub use crate::point::Point3d;
#[doc(inline)]
pub use crate::point::Point4d;

pub mod bbox;
#[doc(inline)]
pub use crate::bbox::bb2d;
#[doc(inline)]
pub use crate::bbox::BBox;
#[doc(inline)]
pub use crate::bbox::BBox2d;
#[doc(inline)]
pub use crate::bbox::BBox3d;
#[doc(inline)]
pub use crate::bbox::BBox4d;

pub mod sym2d;
#[doc(inline)]
pub use crate::sym2d::Sym2d;

pub mod affine_transformation;
#[doc(inline)]
pub use crate::affine_transformation::AffineTransformation;
#[doc(inline)]
pub use crate::affine_transformation::AffineTransformation2d;
#[doc(inline)]
pub use crate::affine_transformation::AffineTransformation3d;
#[doc(inline)]
pub use crate::affine_transformation::AffineTransformation4d;

pub mod array;
#[doc(inline)]
pub use crate::array::Array2d;

mod util;
use crate::util::lex_then;
use crate::util::partial_then;
