// Sketch_AdaptivePredicates.h - Shewchuk-style robust geometric predicates.
//
// Provides adaptive-precision predicates for 2D/3D orientation and incircle
// tests that are guaranteed to return the exact sign for degenerate inputs
// where conventional floating-point evaluation fails.

#pragma once

namespace adaptive_predicate {

// 2D orientation test: sign of the determinant
//   | bx-ax  by-ay |
//   | cx-ax  cy-ay |
// Returns >0 for counter-clockwise, <0 for clockwise, 0 for collinear.
double orient2d(double ax, double ay, double bx, double by, double cx, double cy);

// 3D orientation test: sign of the determinant
//   | bx-ax  by-ay  bz-az |
//   | cx-ax  cy-ay  cz-az |
//   | dx-ax  dy-ay  dz-az |
// Returns >0 if d is below the oriented plane through a,b,c, <0 if above, 0 if coplanar.
double orient3d(double ax, double ay, double az,
                double bx, double by, double bz,
                double cx, double cy, double cz,
                double dx, double dy, double dz);

// 2D incircle test: sign of the determinant
//   | ax ay ax^2+ay^2 1 |
//   | bx by bx^2+by^2 1 |
//   | cx cy cx^2+cy^2 1 |
//   | dx dy dx^2+dy^2 1 |
// Returns >0 if d is inside the circle through a,b,c, <0 if outside, 0 if on the circle.
double incircle(double ax, double ay,
                double bx, double by,
                double cx, double cy,
                double dx, double dy);

} // namespace adaptive_predicate