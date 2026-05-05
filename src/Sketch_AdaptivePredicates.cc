// Sketch_AdaptivePredicates.cc - Shewchuk-style robust geometric predicates.
//
// Implementations of orient2d, orient3d, and incircle using adaptive
// floating-point arithmetic with error-free transformations.

#include "Sketch_AdaptivePredicates.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace adaptive_predicate {

namespace {

// ---- Error-free transformations ------------------------------------------------

// Compute a + b exactly as (sum, error) where sum = a + b (rounded) and
// error = a + b - sum (the floating-point rounding error).
inline void fast_two_sum(double a, double b, double &sum, double &err){
    sum = a + b;
    const double tmp = sum - a;
    err = b - tmp;
}

// Compute a - b exactly.
inline void two_diff(double a, double b, double &diff, double &err){
    diff = a - b;
    const double tmp = diff - a;
    err = a - (diff - tmp) - b;
    // Alternative: err = (a - (diff + b)) + (tmp - b) which is equivalent but
    // more conventional. The form above avoids a double-rounding issue on
    // platforms where the FPU is in double-precision-only mode.
}

// Compute a * b exactly using the two-product algorithm.
inline void two_prod(double a, double b, double &prod, double &err){
    prod = a * b;
    const double a1 = static_cast<double>(static_cast<float>(a));
    const double b1 = static_cast<double>(static_cast<float>(b));
    const double a2 = a - a1;
    const double b2 = b - b1;
    err = (((a1 * b1 - prod) + a1 * b2) + a2 * b1) + a2 * b2;
}

// Split a double into two non-overlapping halves, each with no more than 26
// significant bits.
inline void split(double a, double &hi, double &lo){
    constexpr double splitter = 134217729.0; // 2^27 + 1
    const double tmp = a * splitter;
    hi = tmp - (tmp - a);
    lo = a - hi;
}

// ---- Expansion addition helpers -------------------------------------------------

// Grow an expansion by adding a single value, using linear-time merge.
// The expansion must be a non-overlapping, non-adjacent sequence of doubles
// sorted by increasing magnitude.
template <typename InputIt>
int grow_expansion(int elen, InputIt e_begin, double b, double *h){
    double Q;
    if(b != 0.0){
        fast_two_sum(*e_begin, b, Q, h[0]);
        if(elen > 1){
            ++e_begin;
            for(int i = 1; i < elen; ++i){
                fast_two_sum(*e_begin, Q, Q, h[i]);
                ++e_begin;
            }
            h[elen] = Q;
        }else{
            h[1] = Q;
        }
        return elen + 1;
    }
    for(int i = 0; i < elen; ++i) h[i] = e_begin[i];
    return elen;
}

// Scale an expansion by a constant.
template <typename InputIt>
int scale_expansion(int elen, InputIt e_begin, double b, double *h){
    double Q, sum;
    double hh[8];
    two_prod(*e_begin, b, Q, hh[0]);
    if(elen == 1){
        h[0] = Q;
        h[1] = hh[0];
        return 2;
    }
    double h0[8];
    int hlen = 0;
    int retlen = 0;
    for(int i = 1; i < elen; ++i){
        two_prod(e_begin[i], b, h0[i], h0[i + 1]);
        fast_two_sum(h0[i], Q, sum, hh[hlen]);
        Q = hh[hlen];
        h[retlen] = sum;
        ++retlen;
    }
    h[retlen] = Q;
    ++retlen;
    h[retlen] = h0[elen];
    ++retlen;
    return retlen;
}

// Helper: compress an expansion stored in h[0..hlen-1] in-place.
int compress(int hlen, double *h){
    if(hlen <= 1) return hlen;
    double *bottom = new double[hlen];
    int blen = 0;
    // Top-down pass.
    double Q = h[hlen - 1];
    for(int i = hlen - 2; i >= 0; --i){
        double newQ, err;
        fast_two_sum(Q, h[i], newQ, err);
        if(err != 0.0){
            bottom[blen] = newQ;
            ++blen;
            Q = err;
        }else{
            Q = newQ;
        }
    }
    bottom[blen] = Q;
    ++blen;
    // Bottom-up pass.
    double *top = new double[blen];
    int tlen = 0;
    Q = bottom[0];
    for(int i = 1; i < blen; ++i){
        double newQ, err;
        fast_two_sum(Q, bottom[i], newQ, err);
        if(err != 0.0){
            top[tlen] = newQ;
            ++tlen;
            Q = err;
        }else{
            Q = newQ;
        }
    }
    top[tlen] = Q;
    ++tlen;
    for(int i = 0; i < tlen; ++i) h[i] = top[i];
    delete[] top;
    delete[] bottom;
    return tlen;
}

} // anonymous namespace

// ---- orient2d -------------------------------------------------------------------

double orient2d(double ax, double ay, double bx, double by, double cx, double cy){
    // Compute the exact 2D orientation using Shewchuk's adaptive algorithm.
    const double det_left  = (ax - cx) * (by - cy);
    const double det_right = (ay - cy) * (bx - cx);
    const double det = det_left - det_right;
    double err_bound = 0.0;

    // Estimate the error in the floating-point computation.
    {
        const double abs_det_left  = std::abs(det_left);
        const double abs_det_right = std::abs(det_right);
        const double max_abs = std::max(abs_det_left, abs_det_right);
        const double tol = 3.3306690738754716e-16 * max_abs; // ~3 * epsilon
        err_bound = tol;
    }

    // If the floating-point result is larger than the error bound, it's exact.
    if(std::abs(det) > err_bound) return det;

    // Otherwise, compute exactly using adaptive precision.
    // Compute the exact determinant = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx).
    double u[4];
    double v[4];
    double temp[8];

    // Compute (ax - cx) and (ay - cy) exactly.
    double dx_a, dy_a, dx_e, dy_e;
    two_diff(ax, cx, dx_a, dx_e);
    two_diff(ay, cy, dy_a, dy_e);

    // Compute (bx - cx) and (by - cy) exactly.
    double dx_b, dy_b, dx2_e, dy2_e;
    two_diff(bx, cx, dx_b, dx2_e);
    two_diff(by, cy, dy_b, dy2_e);

    // Compute (ax - cx)*(by - cy) exactly.
    two_prod(dx_a, dy_b, u[0], u[1]);
    // Add (ax - cx)_error * (by - cy) + (ax - cx) * (by - cy)_error.
    if(dx_e != 0.0 || dy2_e != 0.0){
        u[2] = dx_e * dy_b + dx_a * dy2_e;
    }else{
        u[2] = 0.0;
    }

    // Compute (ay - cy)*(bx - cx) exactly.
    two_prod(dy_a, dx_b, v[0], v[1]);
    if(dy_e != 0.0 || dx2_e != 0.0){
        v[2] = dy_e * dx_b + dy_a * dx2_e;
    }else{
        v[2] = 0.0;
    }

    // The expansion lengths.
    int ulen = 3;
    while(ulen > 0 && u[ulen - 1] == 0.0) --ulen;
    int vlen = 3;
    while(vlen > 0 && v[vlen - 1] == 0.0) --vlen;

    if(ulen == 0) return -v[0]; // Only v has non-zero entries.
    if(vlen == 0) return u[0];  // Only u has non-zero entries.

    // Subtract the two expansions: det = u - v.
    // Negate v and add.
    double neg_v[4];
    for(int i = 0; i < vlen; ++i) neg_v[i] = -v[i];

    // Merge expansions u and neg_v.
    double result[8];
    int rlen = 0;
    int ui = 0, vi = 0;
    double r[8];
    // First pass: add the smallest components.
    {
        double Q, err;
        if(std::abs(u[0]) < std::abs(neg_v[0])){
            fast_two_sum(u[0], neg_v[0], Q, err);
            ++ui;
        }else{
            fast_two_sum(neg_v[0], u[0], Q, err);
            ++vi;
        }
        r[0] = err;
        r[1] = Q;
        rlen = 2;
        while(ui < ulen || vi < vlen){
            double next;
            if(vi >= vlen || (ui < ulen && std::abs(u[ui]) < std::abs(neg_v[vi]))){
                next = u[ui];
                ++ui;
            }else{
                next = neg_v[vi];
                ++vi;
            }
            // Add next to expansion r.
            int new_rlen = grow_expansion(rlen, r, next, result);
            for(int i = 0; i < new_rlen; ++i) r[i] = result[i];
            rlen = new_rlen;
        }
    }

    if(rlen == 0) return 0.0;

    // Compress the expansion.
    rlen = compress(rlen, r);

    // Return the sign of the first (largest) component of the compressed expansion.
    // This is guaranteed to have the correct sign if the expansion is exact.
    // For robustness, sum all components.
    double final_result = 0.0;
    for(int i = 0; i < rlen; ++i) final_result += r[i];
    return final_result;
}

// ---- orient3d -------------------------------------------------------------------

double orient3d(double ax, double ay, double az,
                double bx, double by, double bz,
                double cx, double cy, double cz,
                double dx, double dy, double dz){
    // Compute using the standard formula with error estimate, falling back to
    // adaptive precision if the result is not clearly non-zero.
    const double abx = ax - bx;
    const double abz = az - bz;
    const double aby = ay - by;
    const double acx = ax - cx;
    const double acy = ay - cy;
    const double acz = az - cz;
    const double adx = ax - dx;
    const double ady = ay - dy;
    const double adz = az - dz;

    // Det = (abx * (acy * adz - acz * ady) + aby * (acz * adx - acx * adz)
    //        + abz * (acx * ady - acy * adx))
    const double bc_yz = acy * adz - acz * ady;
    const double bc_zx = acz * adx - acx * adz;
    const double bc_xy = acx * ady - acy * adx;

    const double det = abx * bc_yz + aby * bc_zx + abz * bc_xy;

    // Error bound: (|abx|*|bc_yz| + |aby|*|bc_zx| + |abz|*|bc_xy|) * (3*eps)
    const double err_bound = 3.3306690738754716e-16
        * (std::abs(abx) * std::abs(bc_yz)
         + std::abs(aby) * std::abs(bc_zx)
         + std::abs(abz) * std::abs(bc_xy));

    if(std::abs(det) > err_bound) return det;

    // For the fully adaptive case a full expansion-based computation is needed,
    // but it is rarely triggered. Return the float estimate with the note that
    // the result is uncertain.
    // TODO: implement full expansion-based orient3d for degenerate inputs.
    return det;
}

// ---- incircle --------------------------------------------------------------------

double incircle(double ax, double ay,
                double bx, double by,
                double cx, double cy,
                double dx, double dy){
    // Compute using Shewchuk's formula:
    //   | ax ay ax^2+ay^2 1 |
    //   | bx by bx^2+by^2 1 |
    //   | cx cy cx^2+cy^2 1 |
    //   | dx dy dx^2+dy^2 1 |
    //
    // This can be expressed as a linear combination of 4x4 determinants.
    // Using the standard transformed formula:
    // incircle = ((ax-dx)*(ax-dx) + (ay-dy)*(ay-dy)) * orient2d(b,c,d) - ...
    // Actually, compute directly.

    // Shift to place d at the origin:
    const double a_shift_x = ax - dx;
    const double a_shift_y = ay - dy;
    const double b_shift_x = bx - dx;
    const double b_shift_y = by - dy;
    const double c_shift_x = cx - dx;
    const double c_shift_y = cy - dy;

    const double a_sq = a_shift_x * a_shift_x + a_shift_y * a_shift_y;
    const double b_sq = b_shift_x * b_shift_x + b_shift_y * b_shift_y;
    const double c_sq = c_shift_x * c_shift_x + c_shift_y * c_shift_y;

    // Compute orient2d for the shifted points.
    const double o_bc = orient2d(b_shift_x, b_shift_y, c_shift_x, c_shift_y, 0.0, 0.0);
    const double o_ca = orient2d(c_shift_x, c_shift_y, a_shift_x, a_shift_y, 0.0, 0.0);
    const double o_ab = orient2d(a_shift_x, a_shift_y, b_shift_x, b_shift_y, 0.0, 0.0);

    const double det = a_sq * o_bc + b_sq * o_ca + c_sq * o_ab;

    // Error bound using the sum of absolute values.
    const double err_bound = 3.3306690738754716e-16
        * (a_sq * std::abs(o_bc) + b_sq * std::abs(o_ca) + c_sq * std::abs(o_ab));

    if(std::abs(det) > err_bound) return det;

    // TODO: implement fully adaptive incircle for degenerate inputs.
    return det;
}

} // namespace adaptive_predicate