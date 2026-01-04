/**
 * HTTP API server for the Fast MNIST Neural Network.
 * Provides endpoints for digit classification with timing comparison.
 */

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "httplib.h"
#include "json.hpp"

#include "fast_mnist/Matrix.h"
#include "fast_mnist/NeuralNet.h"

using json = nlohmann::json;

/**
 * Baseline (scalar) matrix-vector multiply + sigmoid.
 * This is intentionally NOT optimized to provide a fair comparison.
 */
static void baseline_gemv_sigmoid(const Matrix& W, const Matrix& b,
                                  const std::vector<double>& x,
                                  std::vector<double>& y) {
    const std::size_t m = W.height(), n = W.width();
    for (std::size_t i = 0; i < m; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            s += W[i][k] * x[k];
        }
        s += b[i][0];
        y[i] = 1.0 / (1.0 + std::exp(-s));
    }
}

/**
 * Baseline classify using simple scalar operations.
 * No SIMD, no optimizations - just straightforward C++.
 */
static std::vector<double> baseline_classify(const NeuralNet& net,
                                             const std::vector<double>& input) {
    const auto& weights = net.getWeights();
    const auto& biases = net.getBiases();
    
    // Layer 1: input -> hidden
    std::vector<double> hidden(weights[0].height());
    baseline_gemv_sigmoid(weights[0], biases[0], input, hidden);
    
    // Layer 2: hidden -> output
    std::vector<double> output(weights[1].height());
    baseline_gemv_sigmoid(weights[1], biases[1], hidden, output);
    
    return output;
}

// Model file path
static std::string g_modelPath = "model.weights";

// Load a trained network from file
static NeuralNet loadNetwork(const std::string& path) {
    NeuralNet net({784, 30, 10});
    std::ifstream file(path);
    if (file) {
        file >> net;
        std::cout << "   ✓ Loaded model from " << path << "\n";
    } else {
        std::cout << "   ⚠ No model file found at " << path << "\n";
        std::cout << "     Run: ./fast_mnist_trainer data 10000 5 model.weights\n";
    }
    return net;
}

// Global networks for comparison
static NeuralNet g_network({784, 30, 10});

/**
 * Classify an image using the neural network and return timing info.
 */
json classifyWithTiming(NeuralNet& net, const Matrix& input) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Matrix result = net.classify(input);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);
    
    // Find prediction (max index)
    int prediction = 0;
    double maxVal = result[0][0];
    std::vector<double> confidence(10);
    
    for (int i = 0; i < 10; ++i) {
        confidence[i] = result[i][0];
        if (result[i][0] > maxVal) {
            maxVal = result[i][0];
            prediction = i;
        }
    }
    
    // Normalize confidence to sum to 1 (softmax-like)
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) {
        confidence[i] = std::exp(confidence[i]);
        sum += confidence[i];
    }
    for (int i = 0; i < 10; ++i) {
        confidence[i] /= sum;
    }
    
    return {
        {"prediction", prediction},
        {"confidence", confidence},
        {"time_ms", duration.count()}
    };
}

int main(int argc, char* argv[]) {
    int port = 8080;
    std::string modelPath = "model.weights";
    
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }
    if (argc > 2) {
        modelPath = argv[2];
    }
    
    std::cout << "🧠 Fast MNIST API Server\n";
    std::cout << "   Loading model...\n";
    
    // Load the trained model
    g_network = loadNetwork(modelPath);
    
    httplib::Server svr;
    
    // Enable CORS for frontend
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });
    
    // Handle preflight requests
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });
    
    // Health check endpoint
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"status", "ok"},
            {"version", "1.0.0"}
        };
        res.set_content(response.dump(), "application/json");
    });
    
    // Prediction endpoint
    svr.Post("/predict", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            
            if (!body.contains("pixels") || !body["pixels"].is_array()) {
                res.status = 400;
                res.set_content(R"({"error": "Missing or invalid 'pixels' array"})", 
                               "application/json");
                return;
            }
            
            auto pixels = body["pixels"].get<std::vector<double>>();
            
            if (pixels.size() != 784) {
                res.status = 400;
                std::string error = R"({"error": "Expected 784 pixels, got )" + 
                                   std::to_string(pixels.size()) + "\"}";
                res.set_content(error, "application/json");
                return;
            }
            
            // Create input matrix from pixels
            Matrix input(784, 1, Matrix::NoInit{});
            std::vector<double> inputVec(784);
            for (size_t i = 0; i < 784; ++i) {
                input[i][0] = pixels[i];
                inputVec[i] = pixels[i];
            }
            
            // Run multiple iterations for accurate timing
            const int iterations = 100;
            
            // Baseline classification (scalar, no SIMD)
            auto baselineStart = std::chrono::high_resolution_clock::now();
            std::vector<double> baselineResult;
            for (int iter = 0; iter < iterations; ++iter) {
                baselineResult = baseline_classify(g_network, inputVec);
            }
            auto baselineEnd = std::chrono::high_resolution_clock::now();
            auto baselineTime = std::chrono::duration<double, std::milli>(
                baselineEnd - baselineStart).count() / iterations;
            
            // Optimized classification (SIMD-accelerated)
            auto optimizedStart = std::chrono::high_resolution_clock::now();
            Matrix result;
            for (int iter = 0; iter < iterations; ++iter) {
                result = g_network.classify(input);
            }
            auto optimizedEnd = std::chrono::high_resolution_clock::now();
            auto optimizedTime = std::chrono::duration<double, std::milli>(
                optimizedEnd - optimizedStart).count() / iterations;
            
            // Find prediction from result
            int prediction = 0;
            double maxVal = result[0][0];
            std::vector<double> confidence(10);
            
            for (int i = 0; i < 10; ++i) {
                confidence[i] = result[i][0];
                if (result[i][0] > maxVal) {
                    maxVal = result[i][0];
                    prediction = i;
                }
            }
            
            // Softmax normalization
            double sum = 0.0;
            for (int i = 0; i < 10; ++i) {
                confidence[i] = std::exp(confidence[i]);
                sum += confidence[i];
            }
            for (int i = 0; i < 10; ++i) {
                confidence[i] /= sum;
            }
            
            json response = {
                {"prediction", prediction},
                {"confidence", confidence},
                {"baseline_time_ms", baselineTime},
                {"optimized_time_ms", optimizedTime}
            };
            
            res.set_content(response.dump(), "application/json");
            
        } catch (const json::parse_error& e) {
            res.status = 400;
            json error = {{"error", "Invalid JSON: " + std::string(e.what())}};
            res.set_content(error.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            json error = {{"error", std::string(e.what())}};
            res.set_content(error.dump(), "application/json");
        }
    });
    
    std::cout << "   Listening on http://localhost:" << port << "\n";
    std::cout << "   Endpoints:\n";
    std::cout << "     GET  /health  - Health check\n";
    std::cout << "     POST /predict - Classify a digit\n";
    std::cout << "\n   Press Ctrl+C to stop.\n";
    
    svr.listen("0.0.0.0", port);
    
    return 0;
}
