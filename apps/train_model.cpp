/**
 * Train a neural network on MNIST and save the weights.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "fast_mnist/Matrix.h"
#include "fast_mnist/NeuralNet.h"

namespace fs = std::filesystem;

/**
 * Load a PGM image file into a column vector matrix.
 */
Matrix loadPGM(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to read " + path);
    }

    std::string buf((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
    const char* p = buf.data();
    const char* e = p + buf.size();

    // Skip whitespace and comments
    auto skipSpaceAndComments = [&]() {
        while (p < e) {
            while (p < e && static_cast<unsigned char>(*p) <= ' ')
                ++p;
            if (p < e && *p == '#') {
                while (p < e && *p != '\n')
                    ++p;
                continue;
            }
            break;
        }
    };

    // Parse integer
    auto parseInt = [&]() -> int {
        skipSpaceAndComments();
        int v = 0;
        while (p < e && *p >= '0' && *p <= '9') {
            v = v * 10 + (*p - '0');
            ++p;
        }
        return v;
    };

    skipSpaceAndComments();
    if (p + 2 > e || p[0] != 'P' || p[1] != '2') {
        throw std::runtime_error("Unsupported PGM: " + path);
    }
    p += 2;

    const int width = parseInt();
    const int height = parseInt();
    const int maxVal = parseInt();

    const size_t nPix = static_cast<size_t>(width) * height;
    Matrix img(nPix, 1, Matrix::NoInit{});
    const double inv = 1.0 / static_cast<double>(maxVal);

    for (size_t i = 0; i < nPix; ++i) {
        const int pix = parseInt();
        img[i][0] = static_cast<double>(pix) * inv;
    }

    return img;
}

/**
 * Get expected output from filename (digit is after last underscore).
 */
Matrix getExpectedDigitOutput(const std::string& path) {
    static const std::array<Matrix, 10> oneHot = []() {
        std::array<Matrix, 10> labels;
        for (int d = 0; d < 10; ++d) {
            Matrix m(10, 1, 0.0);
            m[d][0] = 1.0;
            labels[d] = std::move(m);
        }
        return labels;
    }();

    const auto labelPos = path.rfind('_');
    if (labelPos == std::string::npos || labelPos + 1 >= path.size()) {
        throw std::runtime_error("Invalid label path: " + path);
    }
    const int label = path[labelPos + 1] - '0';
    if (label < 0 || label > 9) {
        throw std::runtime_error("Invalid label digit: " + path);
    }
    return oneHot[static_cast<std::size_t>(label)];
}

/**
 * Get the index of maximum element (for classification).
 */
int maxIndex(const Matrix& col) {
    int best = 0;
    double bestVal = col[0][0];
    for (std::size_t r = 1; r < col.height(); ++r) {
        if (col[r][0] > bestVal) {
            bestVal = col[r][0];
            best = static_cast<int>(r);
        }
    }
    return best;
}

/**
 * Assess accuracy on test set.
 */
double assess(NeuralNet& net, const std::string& dataPath,
              const std::vector<std::string>& testFiles) {
    int correct = 0;
    int total = 0;
    for (const auto& imgName : testFiles) {
        fs::path fullPath = fs::path(dataPath) / imgName;
        Matrix img = loadPGM(fullPath.string());
        Matrix exp = getExpectedDigitOutput(imgName);
        Matrix res = net.classify(img);
        if (maxIndex(res) == maxIndex(exp)) {
            ++correct;
        }
        ++total;
    }
    return 100.0 * correct / total;
}

int main(int argc, char* argv[]) {
    std::string dataPath = "data";
    std::string outputPath = "model.weights";
    int imgCount = 60000;
    int epochs = 30;
    int hiddenNeurons = 100;  // Larger hidden layer for better accuracy
    double learningRate = 0.1;  // Smaller rate for online SGD

    if (argc > 1) dataPath = argv[1];
    if (argc > 2) imgCount = std::stoi(argv[2]);
    if (argc > 3) epochs = std::stoi(argv[3]);
    if (argc > 4) hiddenNeurons = std::stoi(argv[4]);
    if (argc > 5) learningRate = std::stod(argv[5]);
    if (argc > 6) outputPath = argv[6];

    std::cout << "Training Configuration:\n";
    std::cout << "  Data path: " << dataPath << "\n";
    std::cout << "  Images: " << imgCount << "\n";
    std::cout << "  Epochs: " << epochs << "\n";
    std::cout << "  Hidden neurons: " << hiddenNeurons << "\n";
    std::cout << "  Learning rate: " << learningRate << "\n";
    std::cout << "  Output: " << outputPath << "\n\n";

    // Load training file list
    std::ifstream fileList("TrainingSetList.txt");
    if (!fileList) {
        throw std::runtime_error("Error reading TrainingSetList.txt");
    }

    std::vector<std::string> trainFiles;
    for (std::string imgName; std::getline(fileList, imgName);) {
        trainFiles.push_back(imgName);
    }

    // Load test file list
    std::ifstream testFileList("TestingSetList.txt");
    std::vector<std::string> testFiles;
    if (testFileList) {
        for (std::string imgName; std::getline(testFileList, imgName);) {
            testFiles.push_back(imgName);
        }
    }

    // Create network with configurable hidden layer
    NeuralNet net({784, hiddenNeurons, 10});

    // Train
    std::random_device rd;
    std::default_random_engine rg(rd());

    double bestAccuracy = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "-- Epoch " << (epoch + 1) << "/" << epochs << " --\n";
        auto start = std::chrono::high_resolution_clock::now();

        std::shuffle(trainFiles.begin(), trainFiles.end(), rg);

        int count = 0;
        for (const auto& imgName : trainFiles) {
            if (count >= imgCount) break;

            fs::path fullPath = fs::path(dataPath) / imgName;
            Matrix img = loadPGM(fullPath.string());
            Matrix exp = getExpectedDigitOutput(imgName);
            net.learn(img, exp, learningRate);
            ++count;

            if (count % 10000 == 0) {
                std::cout << "  Processed " << count << " images\r" << std::flush;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Assess accuracy every epoch
        double accuracy = 0.0;
        if (!testFiles.empty()) {
            accuracy = assess(net, dataPath, testFiles);
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                // Save best model
                std::ofstream bestOut(outputPath);
                if (bestOut) {
                    bestOut << net;
                    bestOut.close();
                }
            }
        }
        
        std::cout << "  Time: " << ms.count() << " ms, Accuracy: " 
                  << std::fixed << std::setprecision(2) << accuracy << "% "
                  << "(best: " << bestAccuracy << "%)\n";
        
        // Early stopping if we reach target accuracy (97%+)
        if (accuracy >= 97.0) {
            std::cout << "\n🎯 Target accuracy of 97% reached!\n";
            break;
        }
    }

    // Final save (best model was already saved during training)
    std::cout << "\n✅ Training complete! Best accuracy: " << std::fixed 
              << std::setprecision(2) << bestAccuracy << "%\n";
    std::cout << "Model saved to " << outputPath << "\n";
    return 0;
}
