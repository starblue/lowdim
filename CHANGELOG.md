# Change Log

## [Unreleased][unreleased]

### Added

### Changed

### Fixed

### Thanks

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
