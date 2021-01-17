//! Affine geometry for discrete 2d to 4d game and puzzle worlds.
//!
//! Provides vectors, matrices, points and more to simplify
//! programming for 2d to 4d games and puzzles.
//! Features include:
//! * Bounding boxes and iterators over the points inside them
//! * Arrays indexed by points
//! * Rotations and reflections
//! * Taxicab (L1) and maximum (L∞) metrics
//! * Orthogonal and king's move neighbourhoods
//!
//! This was originally motivated by
//! [Advent of Code](https://adventofcode.com/).
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
pub use integer::Integer;

pub mod vector;
pub use vector::Vector;
pub use vector::VectorOps;

pub mod vec2d;
pub use crate::vec2d::v2d;
pub use crate::vec2d::Vec2d;

pub mod vec3d;
pub use crate::vec3d::v3d;
pub use crate::vec3d::Vec3d;

pub mod vec4d;
pub use crate::vec4d::v4d;
pub use crate::vec4d::Vec4d;

pub mod matrix2d;
pub use crate::matrix2d::Matrix2d;

pub mod matrix3d;
pub use crate::matrix3d::Matrix3d;

pub mod matrix4d;
pub use crate::matrix4d::Matrix4d;

pub mod point;
pub use crate::point::p2d;
pub use crate::point::p3d;
pub use crate::point::p4d;
pub use crate::point::Point;
pub use crate::point::Point2d;
pub use crate::point::Point3d;
pub use crate::point::Point4d;

pub mod bbox;
pub use crate::bbox::bb2d;
pub use crate::bbox::BBox;
pub use crate::bbox::BBox2d;
pub use crate::bbox::BBox3d;
pub use crate::bbox::BBox4d;

pub mod array;
pub use crate::array::Array2d;

mod util;
use crate::util::lex_then;
use crate::util::partial_then;
