// Property-based tests for the Matrix kernel using rapidcheck.
//
// rapidcheck generates thousands of random inputs per property and
// shrinks failing cases down to a minimal counterexample. We pair it
// with Catch2 via rapidcheck/catch.h so failing properties surface in
// the normal ctest output.
//
// Edge-case floating point is treacherous under generators: if
// rapidcheck happens to draw a NaN or an Inf, most correctness
// properties will fail for reasons that are not actually bugs.
// Every generator in this file clamps its doubles to a finite
// well-behaved range.

#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "fast_mnist/Matrix.h"

namespace {

// Generate a single finite double in [-10, 10]. Tight range keeps
// the accumulator errors small enough that the tolerance checks do
// not need to be adaptive.
rc::Gen<double> boundedDouble() {
    return rc::gen::map(rc::gen::arbitrary<int>(), [](int n) {
        // Map an arbitrary int into a deterministic finite double.
        // Using fmod + normalization avoids ever producing NaN/Inf.
        const double raw = static_cast<double>(n) * 1e-6;
        double v = std::fmod(raw, 20.0);
        if (std::isnan(v) || std::isinf(v))
            v = 0.0;
        return v - 10.0;
    });
}

// Shape generator that keeps matrices small enough for rapidcheck to
// run many iterations per case; large random GEMMs would starve the
// test budget.
struct Shape {
    std::size_t rows;
    std::size_t cols;
};

rc::Gen<Shape> shape(std::size_t maxDim = 8) {
    return rc::gen::construct<Shape>(
        rc::gen::inRange<std::size_t>(1, maxDim + 1),
        rc::gen::inRange<std::size_t>(1, maxDim + 1));
}

// Build a Matrix filled from a flat vector in row-major order. If the
// vector is smaller than rows*cols the remainder is zero-filled; if
// it is larger the tail is discarded.
Matrix makeMatrix(std::size_t rows, std::size_t cols,
                  const std::vector<double>& flat) {
    Matrix m(rows, cols, 0.0);
    std::size_t idx = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            m[r][c] = (idx < flat.size()) ? flat[idx] : 0.0;
            ++idx;
        }
    }
    return m;
}

} // namespace

TEST_CASE("rapidcheck: transpose is an involution",
          "[matrix][property][rapidcheck]") {
    rc::prop("transpose(transpose(A)) == A", []() {
        const Shape s = *shape();
        const auto values =
            *rc::gen::container<std::vector<double>>(s.rows * s.cols,
                                                     boundedDouble());
        Matrix a = makeMatrix(s.rows, s.cols, values);
        Matrix roundTrip = a.transpose().transpose();

        RC_ASSERT(roundTrip.height() == a.height());
        RC_ASSERT(roundTrip.width() == a.width());
        for (std::size_t r = 0; r < a.height(); ++r) {
            for (std::size_t c = 0; c < a.width(); ++c) {
                // Transpose is a pure data move, no FP op: exact.
                RC_ASSERT(roundTrip[r][c] == a[r][c]);
            }
        }
    });
}

TEST_CASE("rapidcheck: axpy with alpha = 0 is a no-op",
          "[matrix][property][rapidcheck]") {
    rc::prop("alpha=0 leaves LHS untouched", []() {
        const Shape s = *shape();
        const auto xVals =
            *rc::gen::container<std::vector<double>>(s.rows * s.cols,
                                                     boundedDouble());
        const auto yVals =
            *rc::gen::container<std::vector<double>>(s.rows * s.cols,
                                                     boundedDouble());
        Matrix x = makeMatrix(s.rows, s.cols, xVals);
        Matrix y = makeMatrix(s.rows, s.cols, yVals);
        Matrix xCopy = x;

        x.axpy(0.0, y);

        for (std::size_t r = 0; r < s.rows; ++r) {
            for (std::size_t c = 0; c < s.cols; ++c) {
                // Exact: axpy with alpha=0 must perform zero FP ops.
                RC_ASSERT(x[r][c] == xCopy[r][c]);
            }
        }
    });
}

TEST_CASE("rapidcheck: matmul shape is preserved",
          "[matrix][property][rapidcheck]") {
    // For A (m x k) and B (k x n), A.dot(B) must have shape m x n.
    rc::prop("shape(A.dot(B)) == (A.rows, B.cols)", []() {
        // Draw three independent dims and compose two matrices whose
        // inner dimension matches by construction.
        const std::size_t m = *rc::gen::inRange<std::size_t>(1, 9);
        const std::size_t k = *rc::gen::inRange<std::size_t>(1, 9);
        const std::size_t n = *rc::gen::inRange<std::size_t>(1, 9);

        const auto lhsVals = *rc::gen::container<std::vector<double>>(
            m * k, boundedDouble());
        const auto rhsVals = *rc::gen::container<std::vector<double>>(
            k * n, boundedDouble());

        Matrix A = makeMatrix(m, k, lhsVals);
        Matrix B = makeMatrix(k, n, rhsVals);

        Matrix C = A.dot(B);
        RC_ASSERT(C.height() == m);
        RC_ASSERT(C.width() == n);
    });
}

TEST_CASE("rapidcheck: dot with zero matrix is zero",
          "[matrix][property][rapidcheck]") {
    rc::prop("A.dot(0) == 0", []() {
        const Shape s = *shape();
        const auto vals = *rc::gen::container<std::vector<double>>(
            s.rows * s.cols, boundedDouble());
        Matrix A = makeMatrix(s.rows, s.cols, vals);
        Matrix Zero(s.cols, s.cols, 0.0);

        Matrix C = A.dot(Zero);
        RC_ASSERT(C.height() == s.rows);
        RC_ASSERT(C.width() == s.cols);
        for (std::size_t r = 0; r < C.height(); ++r) {
            for (std::size_t c = 0; c < C.width(); ++c) {
                // Exact zero: 0 * x + 0 * y + ... is exactly zero in
                // IEEE-754 regardless of x,y, barring NaN/Inf which
                // we've already filtered out in boundedDouble().
                RC_ASSERT(C[r][c] == 0.0);
            }
        }
    });
}

TEST_CASE("rapidcheck: addition is commutative",
          "[matrix][property][rapidcheck]") {
    rc::prop("A + B == B + A", []() {
        const Shape s = *shape();
        const auto va = *rc::gen::container<std::vector<double>>(
            s.rows * s.cols, boundedDouble());
        const auto vb = *rc::gen::container<std::vector<double>>(
            s.rows * s.cols, boundedDouble());
        Matrix A = makeMatrix(s.rows, s.cols, va);
        Matrix B = makeMatrix(s.rows, s.cols, vb);

        Matrix AB = A + B;
        Matrix BA = B + A;
        for (std::size_t r = 0; r < s.rows; ++r) {
            for (std::size_t c = 0; c < s.cols; ++c) {
                // Commutativity of + is exact in IEEE-754 as long as
                // neither operand is NaN.
                RC_ASSERT(AB[r][c] == BA[r][c]);
            }
        }
    });
}
