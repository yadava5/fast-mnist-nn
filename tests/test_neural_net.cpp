#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

#include "fast_mnist/Matrix.h"
#include "fast_mnist/NeuralNet.h"

namespace {

std::string makeZeroNetStream() {
    Matrix layerSizes(1, 3, 0.0);
    layerSizes[0][0] = 2.0;
    layerSizes[0][1] = 2.0;
    layerSizes[0][2] = 2.0;

    Matrix b1(2, 1, 0.0);
    Matrix b2(2, 1, 0.0);
    Matrix w1(2, 2, 0.0);
    Matrix w2(2, 2, 0.0);

    std::ostringstream os;
    os << layerSizes << '\n';
    os << b1 << '\n';
    os << b2 << '\n';
    os << w1 << '\n';
    os << w2 << '\n';
    return os.str();
}

// Build a 2-3-2 network with small hand-picked weights so that the
// forward pass produces predictable, non-trivial activations for
// interpretability tests.
std::string makeTinyNetStream() {
    Matrix layerSizes(1, 3, 0.0);
    layerSizes[0][0] = 2.0;
    layerSizes[0][1] = 3.0;
    layerSizes[0][2] = 2.0;

    // Biases: one column per layer (hidden, output).
    Matrix b1(3, 1, 0.0);
    b1[0][0] = 0.1;
    b1[1][0] = -0.2;
    b1[2][0] = 0.05;

    Matrix b2(2, 1, 0.0);
    b2[0][0] = -0.1;
    b2[1][0] = 0.2;

    // W0: 3x2 (hidden x input).
    Matrix w1(3, 2, 0.0);
    w1[0][0] = 0.5;  w1[0][1] = -0.3;
    w1[1][0] = 0.1;  w1[1][1] = 0.4;
    w1[2][0] = -0.2; w1[2][1] = 0.7;

    // W1: 2x3 (output x hidden).
    Matrix w2(2, 3, 0.0);
    w2[0][0] = 0.3;  w2[0][1] = -0.5; w2[0][2] = 0.2;
    w2[1][0] = -0.1; w2[1][1] = 0.6;  w2[1][2] = 0.4;

    std::ostringstream os;
    os << layerSizes << '\n';
    os << b1 << '\n';
    os << b2 << '\n';
    os << w1 << '\n';
    os << w2 << '\n';
    return os.str();
}

} // namespace

TEST_CASE("NeuralNet loads from stream", "[neural_net]") {
    NeuralNet net({1, 1});
    std::istringstream is(makeZeroNetStream());
    is >> net;

    Matrix input(2, 1, 0.0);
    input[0][0] = 1.0;
    input[1][0] = 1.0;

    Matrix out = net.classify(input);

    REQUIRE(out.height() == 2);
    REQUIRE(out.width() == 1);
    REQUIRE(out[0][0] == Catch::Approx(0.5));
    REQUIRE(out[1][0] == Catch::Approx(0.5));
}

TEST_CASE("classifyWithHidden exposes hidden activations",
          "[neural_net][interpretability]") {
    NeuralNet net({1, 1});
    std::istringstream is(makeTinyNetStream());
    is >> net;

    Matrix input(2, 1, 0.0);
    input[0][0] = 1.0;
    input[1][0] = 0.0;

    std::vector<Val> hidden;
    Matrix out = net.classifyWithHidden(input, hidden);

    // Hidden width matches the middle layer ({2, 3, 2}).
    REQUIRE(hidden.size() == 3);
    for (double h : hidden) {
        REQUIRE(h > 0.0);
        REQUIRE(h < 1.0);
    }

    // Output activations match the plain classify() path, confirming
    // no drift between the two forward paths.
    Matrix refOut = net.classify(input);
    REQUIRE(out.height() == refOut.height());
    REQUIRE(out[0][0] == Catch::Approx(refOut[0][0]));
    REQUIRE(out[1][0] == Catch::Approx(refOut[1][0]));
}

TEST_CASE("loadBinary round-trips a tiny network",
          "[neural_net][binary_weights]") {
    NeuralNet src({1, 1});
    {
        std::istringstream is(makeTinyNetStream());
        is >> src;
    }

    // Manually hand-build the binary payload that export_weights
    // would produce for this network, mirroring the documented
    // layout in include/fast_mnist/NeuralNet.h.
    const std::uint32_t magic = 0x464D4E4Eu;
    const std::uint32_t version = 1u;
    const std::uint32_t layerCount = 3u;
    const std::uint32_t dims[3] = {2u, 3u, 2u};

    std::vector<unsigned char> buf;
    auto appendU32 = [&](std::uint32_t v) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
        buf.insert(buf.end(), p, p + 4);
    };
    auto appendF32 = [&](float v) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
        buf.insert(buf.end(), p, p + 4);
    };

    appendU32(magic);
    appendU32(version);
    appendU32(layerCount);
    for (std::uint32_t d : dims) appendU32(d);

    for (const auto& b : src.getBiases()) {
        for (std::size_t r = 0; r < b.height(); ++r) {
            appendF32(static_cast<float>(b[r][0]));
        }
    }
    for (const auto& w : src.getWeights()) {
        for (std::size_t r = 0; r < w.height(); ++r) {
            for (std::size_t c = 0; c < w.width(); ++c) {
                appendF32(static_cast<float>(w[r][c]));
            }
        }
    }

    NeuralNet restored({1, 1});
    restored.loadBinary(buf.data(), buf.size());

    Matrix input(2, 1, 0.0);
    input[0][0] = 1.0;
    input[1][0] = 0.0;

    Matrix outSrc = src.classify(input);
    Matrix outRestored = restored.classify(input);

    REQUIRE(outRestored.height() == outSrc.height());
    // Loose tolerance -- double -> float32 -> double round-trip
    // drops ~7 decimal digits of precision.
    for (std::size_t i = 0; i < outSrc.height(); ++i) {
        REQUIRE(outRestored[i][0] ==
                Catch::Approx(outSrc[i][0]).margin(1e-5));
    }
}

TEST_CASE("loadBinary rejects bad magic", "[neural_net][binary_weights]") {
    NeuralNet net({1, 1});
    unsigned char bad[12] = {0};
    REQUIRE_THROWS(net.loadBinary(bad, sizeof(bad)));
}

TEST_CASE("computeInputGradient returns input-sized saliency",
          "[neural_net][interpretability]") {
    NeuralNet net({1, 1});
    std::istringstream is(makeTinyNetStream());
    is >> net;

    Matrix input(2, 1, 0.0);
    input[0][0] = 1.0;
    input[1][0] = 0.0;

    std::vector<Val> grad;
    net.computeInputGradient(input, 0, grad);

    REQUIRE(grad.size() == 2);

    // For this hand-picked network the gradient must be non-zero in
    // at least one component -- a trivial zero response would mean
    // backprop is dropping the signal.
    const bool anyNonZero =
        std::abs(grad[0]) > 1e-12 || std::abs(grad[1]) > 1e-12;
    REQUIRE(anyNonZero);

    // Finite-difference sanity check: perturb each input pixel and
    // compare (f(x+e) - f(x)) / eps against the analytic gradient
    // of the class-0 output activation.
    const double eps = 1e-6;
    Matrix baseOut = net.classify(input);
    for (int i = 0; i < 2; ++i) {
        Matrix perturbed(2, 1, 0.0);
        perturbed[0][0] = input[0][0];
        perturbed[1][0] = input[1][0];
        perturbed[i][0] += eps;
        Matrix pOut = net.classify(perturbed);
        const double fdGrad = (pOut[0][0] - baseOut[0][0]) / eps;
        // Loose tolerance -- finite differences at 1e-6 on a
        // sigmoid stack are typically accurate to ~1e-4.
        REQUIRE(std::abs(grad[i] - fdGrad) < 5e-4);
    }
}
