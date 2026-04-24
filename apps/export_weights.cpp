/**
 * Convert a trained ASCII model.weights file into a compact little-
 * endian binary (.bin) blob suitable for streaming to the browser and
 * loading through NeuralNet::loadBinary.
 *
 * File layout (all little-endian):
 *
 *   uint32_t magic       = 'FMNN' (0x464D4E4E)
 *   uint32_t version     = 1
 *   uint32_t layerCount
 *   uint32_t layerSizes[layerCount]
 *   float32  biases_l0 [layerSizes[1]]
 *   float32  biases_l1 [layerSizes[2]]
 *   ...
 *   float32  weights_l0[layerSizes[1] * layerSizes[0]]
 *   float32  weights_l1[layerSizes[2] * layerSizes[1]]
 *   ...
 *
 * Casting double -> float32 halves the payload with no observable
 * accuracy impact on the MNIST MLP (sigmoids saturate well before
 * f32 rounding error matters). The network can always be retrained
 * and re-exported if precision becomes an issue.
 *
 * Usage: export_weights <input_ascii> <output_binary>
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fast_mnist/Matrix.h"
#include "fast_mnist/NeuralNet.h"

namespace {

void writeU32(std::ofstream& out, std::uint32_t v) {
    unsigned char bytes[4];
    std::memcpy(bytes, &v, sizeof(v));
    out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void writeF32(std::ofstream& out, float v) {
    unsigned char bytes[4];
    std::memcpy(bytes, &v, sizeof(v));
    out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_ascii_weights> <output_binary_weights>\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];

    // The NeuralNet operator>> needs a target with a layer topology.
    // We pass something plausible; the stream reader rebuilds the
    // actual topology from the ASCII payload.
    NeuralNet net({784, 100, 10});
    {
        std::ifstream in(inputPath);
        if (!in) {
            std::cerr << "Error: cannot open " << inputPath << "\n";
            return 1;
        }
        in >> net;
        if (!in) {
            std::cerr << "Error: failed to parse " << inputPath << "\n";
            return 1;
        }
    }

    const auto& biases = net.getBiases();
    const auto& weights = net.getWeights();
    if (biases.size() != weights.size() || biases.empty()) {
        std::cerr << "Error: parsed network has no layers\n";
        return 1;
    }

    // Derive layer sizes from the weight matrices. For a network
    // with weight layout [L1,L0], [L2,L1], ..., the layer sizes are
    // [weights[0].width, weights[0].height, weights[1].height, ...].
    std::vector<std::uint32_t> dims;
    dims.reserve(weights.size() + 1);
    dims.push_back(static_cast<std::uint32_t>(weights.front().width()));
    for (const auto& w : weights) {
        dims.push_back(static_cast<std::uint32_t>(w.height()));
    }

    std::ofstream out(outputPath, std::ios::binary);
    if (!out) {
        std::cerr << "Error: cannot open " << outputPath << " for writing\n";
        return 1;
    }

    writeU32(out, 0x464D4E4Eu); // 'FMNN'
    writeU32(out, 1u);           // version
    writeU32(out, static_cast<std::uint32_t>(dims.size()));
    for (std::uint32_t d : dims) {
        writeU32(out, d);
    }

    // Biases: layer by layer.
    for (const auto& b : biases) {
        for (std::size_t r = 0; r < b.height(); ++r) {
            writeF32(out, static_cast<float>(b[r][0]));
        }
    }

    // Weights: row-major per layer.
    for (const auto& w : weights) {
        for (std::size_t r = 0; r < w.height(); ++r) {
            for (std::size_t c = 0; c < w.width(); ++c) {
                writeF32(out, static_cast<float>(w[r][c]));
            }
        }
    }

    if (!out) {
        std::cerr << "Error: write failed\n";
        return 1;
    }
    out.close();

    std::cout << "Wrote " << outputPath << " (" << dims.size()
              << " layers: ";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        std::cout << dims[i] << (i + 1 < dims.size() ? " -> " : "");
    }
    std::cout << ")\n";
    return 0;
}
