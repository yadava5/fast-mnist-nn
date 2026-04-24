/**
 * Emscripten / Embind bindings exposing NeuralNet classification to
 * JavaScript. Built only when CMake is configured with an Emscripten
 * toolchain (see the top-level CMakeLists.txt).
 *
 * Design notes:
 *   - Weights arrive as a Uint8Array from fetch('model.weights.bin').
 *     We copy the bytes into a std::string container because Embind's
 *     val<->vector<uint8_t> path is awkward; std::string roundtrips
 *     cleanly and is used as an opaque byte buffer here.
 *   - Pixels arrive as a plain Array<number> or Float32Array. We read
 *     them through val and convert to Matrix's double storage.
 *   - classify() returns a plain JS object matching the existing
 *     PredictionResponse shape (prediction, confidence, hidden,
 *     input_grad) so the TS fallback glue is near-trivial.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "fast_mnist/Matrix.h"
#include "fast_mnist/NeuralNet.h"

namespace {

using emscripten::val;

// Helper: convert a JS array-like (Array, Float32Array, Uint8Array of
// numbers ...) into a contiguous std::vector<double>. Uses the generic
// val::as<std::vector<T>>() path which Embind wires for all
// number-like typed arrays.
std::vector<double> jsToPixels(const val& pixelsJs) {
    // The VectorFromJSArray helper returned by Embind accepts both
    // plain Array and typed arrays. It copies into a new vector of
    // doubles; for the 784-pixel path this is a negligible one-time
    // copy.
    return emscripten::vecFromJSArray<double>(pixelsJs);
}

val toFloatArray(const std::vector<double>& src) {
    val arr = val::array();
    for (std::size_t i = 0; i < src.size(); ++i) {
        arr.set(i, src[i]);
    }
    return arr;
}

} // namespace

/**
 * Thin JS-facing shim around a NeuralNet. The class is default
 * constructed with the canonical 784->100->10 topology; loadBinary
 * then rebuilds the layer sizes, biases, and weights to match the
 * payload. This keeps construction cheap (no weight init) and lets
 * the JS side treat model loading as an async one-shot step.
 */
class WasmClassifier {
  public:
    WasmClassifier() : net_({784, 100, 10}) {}

    /**
     * Replace the network's weights and biases with the contents of
     * the supplied binary payload. The string is used purely as a
     * byte container (Embind marshals JS Uint8Array to std::string
     * cleanly when the string overload is selected).
     */
    void loadWeightsFromBinary(const std::string& bytes) {
        if (bytes.empty()) {
            throw std::runtime_error("weights buffer is empty");
        }
        net_.loadBinary(
            reinterpret_cast<const unsigned char*>(bytes.data()),
            bytes.size());
    }

    /**
     * Run forward inference on a 784-long pixel vector and return a
     * JS object matching the HTTP /predict response shape.
     */
    val classify(const val& pixelsJs) {
        std::vector<double> pixels = jsToPixels(pixelsJs);
        if (pixels.size() != 784) {
            throw std::runtime_error(
                "expected 784 pixels, got " + std::to_string(pixels.size()));
        }

        Matrix input(784, 1, Matrix::NoInit{});
        for (std::size_t i = 0; i < 784; ++i) {
            input[i][0] = pixels[i];
        }

        // Forward pass with hidden-layer capture.
        std::vector<double> hidden;
        Matrix logits = net_.classifyWithHidden(input, hidden);

        // argmax + softmax-style normalization (matches server.cpp's
        // response shape so the TS layer can treat both sources
        // interchangeably).
        std::vector<double> confidence(logits.height());
        int prediction = 0;
        double maxVal = logits[0][0];
        for (std::size_t i = 0; i < logits.height(); ++i) {
            confidence[i] = logits[i][0];
            if (logits[i][0] > maxVal) {
                maxVal = logits[i][0];
                prediction = static_cast<int>(i);
            }
        }
        double sum = 0.0;
        for (std::size_t i = 0; i < confidence.size(); ++i) {
            confidence[i] = std::exp(confidence[i]);
            sum += confidence[i];
        }
        if (sum > 0.0) {
            for (double& c : confidence) c /= sum;
        }

        // Saliency: gradient of the predicted-class activation wrt
        // the input pixels. Single pass, not timed -- the UI treats
        // this as an interpretability bonus, not part of the hot
        // inference loop.
        std::vector<double> inputGrad;
        net_.computeInputGradient(input, prediction, inputGrad);

        val result = val::object();
        result.set("prediction", prediction);
        result.set("confidence", toFloatArray(confidence));
        result.set("hidden_activations", toFloatArray(hidden));
        result.set("input_grad", toFloatArray(inputGrad));
        return result;
    }

  private:
    NeuralNet net_;
};

EMSCRIPTEN_BINDINGS(fast_mnist) {
    emscripten::class_<WasmClassifier>("WasmClassifier")
        .constructor<>()
        .function("loadWeightsFromBinary",
                  &WasmClassifier::loadWeightsFromBinary)
        .function("classify", &WasmClassifier::classify);
}
