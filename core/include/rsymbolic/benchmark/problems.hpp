#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"

namespace rsymbolic {

// A self-contained benchmark problem. Data is generated programmatically (no external
// files), so the benchmark suite works on any platform without extra setup.
//
// Recovery is judged on X_test/y_test (held-out + extrapolation points) rather than
// the training loss, so over-fitted approximations are classified as "failed" rather
// than "recovered".
struct BenchmarkProblem {
    std::string name;
    std::vector<std::vector<double>> X;       // training inputs (20 points)
    std::vector<double> y;                    // training targets
    std::vector<std::vector<double>> X_test;  // held-out + extrapolation inputs
    std::vector<double> y_test;               // held-out targets
    SearchSpace space;
    // max |f(x_test) - y_test| below this => "recovered".
    // Proxy for ground-truth structural recovery: an over-fitted approximation
    // will diverge on extrapolation points even when train loss is small.
    double recovery_threshold = 1e-4;
};

// Generate evenly spaced points x in [lo, hi].
inline std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        v[static_cast<std::size_t>(i)] =
            lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return v;
}

// Populate training and test data from a scalar function.
// xs_train and xs_test are the 1-D input grids (single-feature problems).
inline void fill_1d(BenchmarkProblem& p,
                    const std::function<double(double)>& f,
                    const std::vector<double>& xs_train,
                    const std::vector<double>& xs_test) {
    p.X.clear();
    p.y.clear();
    for (double x : xs_train) {
        p.X.push_back({x});
        p.y.push_back(f(x));
    }
    p.X_test.clear();
    p.y_test.clear();
    for (double x : xs_test) {
        p.X_test.push_back({x});
        p.y_test.push_back(f(x));
    }
}

// Populate training and test data from a multi-feature function.
// rows_train / rows_test are pre-built input row vectors.
inline void fill_rows(BenchmarkProblem& p,
                      const std::function<double(const std::vector<double>&)>& f,
                      std::vector<std::vector<double>> rows_train,
                      std::vector<std::vector<double>> rows_test) {
    p.y.resize(rows_train.size());
    for (std::size_t i = 0; i < rows_train.size(); ++i)
        p.y[i] = f(rows_train[i]);
    p.X = std::move(rows_train);

    p.y_test.resize(rows_test.size());
    for (std::size_t i = 0; i < rows_test.size(); ++i)
        p.y_test[i] = f(rows_test[i]);
    p.X_test = std::move(rows_test);
}

// Generate an n×n grid of 2-D input rows covering [lo, hi] × [lo, hi].
inline std::vector<std::vector<double>> grid2d(double lo, double hi, int n) {
    const auto xs = linspace(lo, hi, n);
    std::vector<std::vector<double>> rows;
    rows.reserve(static_cast<std::size_t>(n * n));
    for (double x0 : xs)
        for (double x1 : xs)
            rows.push_back({x0, x1});
    return rows;
}

// Baseline: y = 2.5*x + 1.7  (linear, trivially expressible)
// Test range extends to [-8, 8] to expose structural mis-matches via extrapolation.
inline BenchmarkProblem problem_linear() {
    BenchmarkProblem p;
    p.name = "linear (2.5x+1.7)";
    const auto f = [](double x) { return 2.5 * x + 1.7; };
    fill_1d(p, f, linspace(-5.0, 5.0, 20), linspace(-8.0, 8.0, 40));
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {};
    p.space.num_features = 1;
    p.space.max_depth = 3;
    p.space.max_nodes = 30;
    return p;
}

// Baseline: y = 2*exp(0.3*x)  (single-variable exponential)
// Test range extends to [0, 4] to catch wrong-structure approximations.
inline BenchmarkProblem problem_exponential() {
    BenchmarkProblem p;
    p.name = "exponential (2*exp(0.3x))";
    const auto f = [](double x) { return 2.0 * std::exp(0.3 * x); };
    fill_1d(p, f, linspace(0.0, 3.0, 20), linspace(0.0, 4.0, 40));
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {UnaryOp::Exp};
    p.space.num_features = 1;
    p.space.max_depth = 4;
    p.space.max_nodes = 30;
    return p;
}

// Nguyen-1: y = x^3 + x^2 + x  (polynomial; needs structural depth to find x*x*x)
// Test range extends to [-1.5, 1.5].
inline BenchmarkProblem problem_nguyen1() {
    BenchmarkProblem p;
    p.name = "Nguyen-1 (x^3+x^2+x)";
    const auto f = [](double x) { return x * x * x + x * x + x; };
    fill_1d(p, f, linspace(-1.0, 1.0, 20), linspace(-1.5, 1.5, 40));
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {UnaryOp::Neg};
    p.space.num_features = 1;
    p.space.max_depth = 5;
    p.space.max_nodes = 40;
    return p;
}

// Nguyen-5: y = sin(x^2)*cos(x) - 1  (requires both sin and cos)
// Test range extends to [-1.5, 1.5].
inline BenchmarkProblem problem_nguyen5() {
    BenchmarkProblem p;
    p.name = "Nguyen-5 (sin(x^2)*cos(x)-1)";
    const auto f = [](double x) { return std::sin(x * x) * std::cos(x) - 1.0; };
    fill_1d(p, f, linspace(-1.0, 1.0, 20), linspace(-1.5, 1.5, 40));
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Neg};
    p.space.num_features = 1;
    p.space.max_depth = 5;
    p.space.max_nodes = 40;
    return p;
}

// Nguyen-7: y = log(x+1) + log(x^2+1)  (requires log; x > 0 avoids domain issues)
// Test range extends to [0, 3].
inline BenchmarkProblem problem_nguyen7() {
    BenchmarkProblem p;
    p.name = "Nguyen-7 (log(x+1)+log(x^2+1))";
    const auto f = [](double x) { return std::log(x + 1.0) + std::log(x * x + 1.0); };
    fill_1d(p, f, linspace(0.0, 2.0, 20), linspace(0.0, 3.0, 40));
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {UnaryOp::Log, UnaryOp::Neg};
    p.space.num_features = 1;
    p.space.max_depth = 5;
    p.space.max_nodes = 40;
    return p;
}

// Two-feature problem: y = x0*x1 + x0  (verifies multi-variable search path).
// Training: 5×5 grid on [-2, 2]; test: 7×7 grid on [-3, 3] (extrapolation).
inline BenchmarkProblem problem_multivar() {
    BenchmarkProblem p;
    p.name = "multivar (x0*x1 + x0)";
    const auto f = [](const std::vector<double>& x) { return x[0] * x[1] + x[0]; };
    fill_rows(p, f, grid2d(-2.0, 2.0, 5), grid2d(-3.0, 3.0, 7));
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {};
    p.space.num_features = 2;
    p.space.max_depth = 4;
    p.space.max_nodes = 30;
    return p;
}

// Return all standard problems in difficulty order.
inline std::vector<BenchmarkProblem> standard_problems() {
    return {problem_linear(), problem_exponential(), problem_nguyen1(),
            problem_nguyen5(), problem_nguyen7(), problem_multivar()};
}

}  // namespace rsymbolic
