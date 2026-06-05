#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"

namespace rsymbolic {

// A self-contained benchmark problem. Data is generated programmatically (no external
// files), so the benchmark suite works on any platform without extra setup.
struct BenchmarkProblem {
    std::string name;
    std::vector<std::vector<double>> X;  // input rows
    std::vector<double> y;               // target values
    SearchSpace space;                   // suggested operator set for this problem
    double recovery_threshold = 1e-6;    // loss < this => "recovered"
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

// Baseline: y = 2.5*x + 1.7  (linear, trivially expressible)
inline BenchmarkProblem problem_linear() {
    BenchmarkProblem p;
    p.name = "linear (2.5x+1.7)";
    for (const double x : linspace(-5.0, 5.0, 20)) {
        p.X.push_back({x});
        p.y.push_back(2.5 * x + 1.7);
    }
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {};
    p.space.num_features = 1;
    p.space.max_depth = 3;
    p.space.max_nodes = 30;
    return p;
}

// Baseline: y = 2*exp(0.3*x)  (single-variable exponential)
inline BenchmarkProblem problem_exponential() {
    BenchmarkProblem p;
    p.name = "exponential (2*exp(0.3x))";
    for (const double x : linspace(0.0, 3.0, 20)) {
        p.X.push_back({x});
        p.y.push_back(2.0 * std::exp(0.3 * x));
    }
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {UnaryOp::Exp};
    p.space.num_features = 1;
    p.space.max_depth = 4;
    p.space.max_nodes = 30;
    return p;
}

// Nguyen-1: y = x^3 + x^2 + x  (polynomial; needs structural depth to find x*x*x)
inline BenchmarkProblem problem_nguyen1() {
    BenchmarkProblem p;
    p.name = "Nguyen-1 (x^3+x^2+x)";
    for (const double x : linspace(-1.0, 1.0, 20)) {
        p.X.push_back({x});
        p.y.push_back(x * x * x + x * x + x);
    }
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {UnaryOp::Neg};
    p.space.num_features = 1;
    p.space.max_depth = 5;
    p.space.max_nodes = 40;
    return p;
}

// Nguyen-5: y = sin(x^2)*cos(x) - 1  (requires both sin and cos)
inline BenchmarkProblem problem_nguyen5() {
    BenchmarkProblem p;
    p.name = "Nguyen-5 (sin(x^2)*cos(x)-1)";
    for (const double x : linspace(-1.0, 1.0, 20)) {
        p.X.push_back({x});
        p.y.push_back(std::sin(x * x) * std::cos(x) - 1.0);
    }
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Neg};
    p.space.num_features = 1;
    p.space.max_depth = 5;
    p.space.max_nodes = 40;
    return p;
}

// Nguyen-7: y = log(x+1) + log(x^2+1)  (requires log; x > 0 avoids domain issues)
inline BenchmarkProblem problem_nguyen7() {
    BenchmarkProblem p;
    p.name = "Nguyen-7 (log(x+1)+log(x^2+1))";
    for (const double x : linspace(0.0, 2.0, 20)) {
        p.X.push_back({x});
        p.y.push_back(std::log(x + 1.0) + std::log(x * x + 1.0));
    }
    p.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    p.space.unary_ops = {UnaryOp::Log, UnaryOp::Neg};
    p.space.num_features = 1;
    p.space.max_depth = 5;
    p.space.max_nodes = 40;
    return p;
}

// Return all standard problems in difficulty order.
inline std::vector<BenchmarkProblem> standard_problems() {
    return {problem_linear(), problem_exponential(), problem_nguyen1(),
            problem_nguyen5(), problem_nguyen7()};
}

}  // namespace rsymbolic
