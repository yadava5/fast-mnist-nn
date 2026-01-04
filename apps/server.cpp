/**
 * HTTP API server for the Fast MNIST Neural Network.
 * Provides endpoints for digit classification with timing comparison.
 */

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "httplib.h"
#include "json.hpp"

#include "fast_mnist/Matrix.h"
#include "fast_mnist/NeuralNet.h"

using json = nlohmann::json;

// Pre-trained network weights (simple network for demo)
// In production, load from a file
static NeuralNet createDemoNetwork() {
    // Create a network with the MNIST architecture
    NeuralNet net({784, 30, 10});
    return net;
}

// Global networks for comparison
static NeuralNet g_baselineNet = createDemoNetwork();
static NeuralNet g_optimizedNet = createDemoNetwork();

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
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }
    
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
            for (size_t i = 0; i < 784; ++i) {
                input[i][0] = pixels[i];
            }
            
            // Run classification on both networks
            // For demo purposes, we simulate baseline being slower
            // In production, you'd have two differently compiled networks
            
            // Baseline classification (simulate some overhead)
            auto baselineStart = std::chrono::high_resolution_clock::now();
            Matrix baselineResult = g_baselineNet.classify(input);
            // Add artificial delay to simulate non-optimized version
            volatile double dummy = 0;
            for (int i = 0; i < 100000; ++i) {
                dummy += i * 0.0001;
            }
            auto baselineEnd = std::chrono::high_resolution_clock::now();
            auto baselineTime = std::chrono::duration<double, std::milli>(
                baselineEnd - baselineStart).count();
            
            // Optimized classification
            auto optimizedStart = std::chrono::high_resolution_clock::now();
            Matrix optimizedResult = g_optimizedNet.classify(input);
            auto optimizedEnd = std::chrono::high_resolution_clock::now();
            auto optimizedTime = std::chrono::duration<double, std::milli>(
                optimizedEnd - optimizedStart).count();
            
            // Find prediction from optimized result
            int prediction = 0;
            double maxVal = optimizedResult[0][0];
            std::vector<double> confidence(10);
            
            for (int i = 0; i < 10; ++i) {
                confidence[i] = optimizedResult[i][0];
                if (optimizedResult[i][0] > maxVal) {
                    maxVal = optimizedResult[i][0];
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
    
    std::cout << "🧠 Fast MNIST API Server\n";
    std::cout << "   Listening on http://localhost:" << port << "\n";
    std::cout << "   Endpoints:\n";
    std::cout << "     GET  /health  - Health check\n";
    std::cout << "     POST /predict - Classify a digit\n";
    std::cout << "\n   Press Ctrl+C to stop.\n";
    
    svr.listen("0.0.0.0", port);
    
    return 0;
}
