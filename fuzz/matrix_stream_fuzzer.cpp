#include "fast_mnist/Matrix.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <ios>
#include <sstream>

namespace {

class FuzzBytes {
  public:
    FuzzBytes(const std::uint8_t* data, std::size_t size)
        : data_(data), size_(size) {}

    std::uint8_t next() {
        if (size_ == 0) {
            return 0;
        }
        const std::uint8_t value = data_[offset_ % size_];
        ++offset_;
        return value;
    }

    double nextValue() {
        std::uint64_t bits = 0;
        for (int i = 0; i < 8; ++i) {
            bits = (bits << 8) | next();
        }

        double value = 0.0;
        static_assert(sizeof(value) == sizeof(bits),
                      "double and uint64_t must have equal size");
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) {
            const int centered = static_cast<int>(bits % 2049) - 1024;
            return static_cast<double>(centered) / 16.0;
        }
        return std::clamp(value, -1024.0, 1024.0);
    }

  private:
    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t offset_{0};
};

void consumeMatrix(const Matrix& matrix) {
    double sum = 0.0;
    for (std::size_t r = 0; r < matrix.height(); ++r) {
        for (std::size_t c = 0; c < matrix.width(); ++c) {
            sum += matrix[r][c] * static_cast<double>(1 + r + c);
        }
    }

    volatile double sink = sum;
    (void)sink;
}

std::string buildMatrixInput(FuzzBytes& bytes) {
    constexpr std::size_t MaxDimension = 8;
    const std::size_t rows = bytes.next() % (MaxDimension + 1);
    const std::size_t cols = bytes.next() % (MaxDimension + 1);

    std::ostringstream stream;
    stream << rows << ' ' << cols << '\n';
    stream << std::setprecision(17);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if ((bytes.next() & 0x3f) == 0) {
                stream << "invalid ";
            } else {
                stream << bytes.nextValue() << ' ';
            }
        }
        stream << '\n';
    }

    return stream.str();
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
    if (size == 0) {
        return 0;
    }

    FuzzBytes bytes(data, size);
    Matrix matrix;
    std::istringstream input(buildMatrixInput(bytes));
    input >> matrix;

    Matrix roundTrip = matrix.transpose().transpose();
    Matrix scaled = roundTrip * 0.25;
    Matrix combined = scaled + matrix;
    combined.axpy(-0.25, matrix);

    if (matrix.width() > 0) {
        Matrix weights(matrix.width(), 1, 1.0);
        consumeMatrix(matrix.dot(weights));
    }

    std::ostringstream output;
    output << combined;
    Matrix reparsed;
    std::istringstream reparsedInput(output.str());
    reparsedInput >> reparsed;

    consumeMatrix(combined);
    consumeMatrix(reparsed);
    return 0;
}
