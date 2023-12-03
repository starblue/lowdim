# Change Log

## [Unreleased] - [unreleased]

### Breaking Changes

### Additions

### Changes

### Fixes

### Thanks

## 0.7.0 - 2023-12-03

### Breaking Changes
- Rename the existing `BBox::extend_to` to `BBox::extended_to` to make room for
  a new method `BBox::extend_to` which modifies the `BBox` in place.

## 0.6.1 - 2023-04-10

### Changes
- Change `Array::with` to take an `FnMut` instead of an `Fn`.
  This allows the function to have internal state that is mutable,
  which is needed when using a random number generator.

## 0.6.0 - 2021-12-30

### Breaking Changes
- Change vector division to `div_euclid` and add remainder via `rem_euclid`.
  This changes the rounding of vector division, and as a consequence also
  the result of `BBox::center`, if the center has some negative coordinate and 
  that coordinate needs rounding.

### Additions
- Add componentwise division and remainder for vectors,
  also via `div_euclid` and `rem_euclid`.
- Implement division and remainder for a point divided by a bounding box
  to support toroidal manifolds,  where objects leaving the bounding box
  at the maximum for a coordinate reenter at the minimum or vice versa.
- Add `Point::distance_l2_squared`.

## 0.5.4 - 2021-12-26

### Changed
- Deprecate `BBox::from_point`, use `BBox::from` instead.

### Added
- Implement `From<Point<S, V>>` for bounding boxes.
  A point is converted to a bounding box containing that single point.
- Implement `Eq` and `Hash` for arrays.

## 0.5.3 - 2021-12-22

### Added
- Intersection for bounding boxes.

## 0.5.2 - 2021-12-13

### Changed
- Rename `BBox::from_points` to `BBox::from_corners` for clarity.
  Keep deprecated `BBox::from_points` for now.

### Added
- `BBox::enclosing` for the smallest bounding box enclosing some points.

## 0.5.1 - 2021-12-11

### Changed
- Update documentation.
- Implement IntoIterator for owned bounding boxes.

## 0.5.0 - 2021-12-10

The methods for generating unit vectors or neighbors of a point
now return iterators instead of vectors (breaking change).
