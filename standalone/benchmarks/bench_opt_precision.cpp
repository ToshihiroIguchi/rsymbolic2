// Evaluation harness for the two user-requested "implement and measure" changes:
//   (1) BFGS  — an alternative constant optimizer to the shipped self-LM (PySR uses BFGS).
//   (2) Float32 — running the constant fit in single precision (PySR precision=32).
//
// Both are listed in CLAUDE.md as ALLOWED DIVERGENCES (implementation method may differ),
// not parity requirements, and docs/32 recommended against shipping either. This harness
// does not change the shipped engine; it implements both as templated standalone solvers
// driven by the PRODUCTION evaluator (tree.hpp evaluate<T>) so the measurement is faithful,
// then reports per-fit wall time AND solution quality (final SSE, max constant error) for
// all four cells: {self-LM, BFGS} x {Float64, Float32}.
//
// Why a harness and not an engine change: the shipped ConstantOptimizer/OptimizationProblem
// interface is hard-locked to std::vector<double>, and the AD types (Dual/MultiDual) are
// double-only. Templating the entire engine + R bindings to float is disproportionate to a
// measurement; the fit is ~93% of search compute (docs/30), so a faithful fit-level
// measurement answers the speed question. The algorithms here are real (not stubs): the LM
// mirrors self_lm_optimizer.cpp and the BFGS is a textbook inverse-Hessian BFGS with
// backtracking Armijo line search (the Optim.jl/PySR default), both with the SR.jl
// multi-start. The same analytic Jacobian (k forward-mode dual passes) feeds both, so the
// LM-vs-BFGS and f64-vs-f32 ratios are apples-to-apples.
//
// Build (Rtools g++):
//   g++ -std=c++17 -O2        -I r-package/rsymbolic2/src \
//       standalone/benchmarks/bench_opt_precision.cpp -o bench_opt_O2
//   g++ -std=c++17 -O3 -march=native -I r-package/rsymbolic2/src \
//       standalone/benchmarks/bench_opt_precision.cpp -o bench_opt_O3
//
// NOT part of the shipped package; pure evidence-gathering.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace rsymbolic {
// Extra value-path overload so evaluate<float> computes square in float (the shipped
// dual.hpp only defines square(double) / square(Dual)). Without this, ADL would promote
// to square(double) and the f32 value path would do that one op in double — we want a
// faithful all-float value path. Must be declared before tree.hpp is included so the
// template instantiation of apply_unary<float> sees it.
inline float square(float a) { return a * a; }
inline float recip(float a) { return 1.0f / a; }  // same reason, for UnaryOp::Inv
}  // namespace rsymbolic

#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// ---------------------------------------------------------------------------------
// Templated forward-mode dual number GDual<T> (T = float or double). Mirrors dual.hpp
// exactly but parameterised on the scalar type, so evaluate<GDual<T>>() produces the
// value + one directional derivative in precision T. Defined in namespace rsymbolic so
// detail::apply_unary/apply_binary find these via ADL. Distinct type from the shipped
// Dual (which is double-only), so no clash.
// ---------------------------------------------------------------------------------
template <typename T>
struct GDual {
    T value{};
    T deriv{};
    GDual() = default;
    GDual(double v) : value(static_cast<T>(v)), deriv(T(0)) {}  // NOLINT: row/literal
    GDual(T v, T d) : value(v), deriv(d) {}
};

template <typename T> GDual<T> operator+(const GDual<T>& a, const GDual<T>& b) {
    return {a.value + b.value, a.deriv + b.deriv};
}
template <typename T> GDual<T> operator-(const GDual<T>& a, const GDual<T>& b) {
    return {a.value - b.value, a.deriv - b.deriv};
}
template <typename T> GDual<T> operator-(const GDual<T>& a) { return {-a.value, -a.deriv}; }
template <typename T> GDual<T> operator*(const GDual<T>& a, const GDual<T>& b) {
    return {a.value * b.value, a.deriv * b.value + a.value * b.deriv};
}
template <typename T> GDual<T> operator/(const GDual<T>& a, const GDual<T>& b) {
    const T inv = T(1) / b.value;
    return {a.value * inv, (a.deriv * b.value - a.value * b.deriv) * inv * inv};
}
template <typename T> GDual<T> exp(const GDual<T>& a) {
    const T e = std::exp(a.value);
    return {e, a.deriv * e};
}
template <typename T> GDual<T> log(const GDual<T>& a) {
    return {std::log(a.value), a.deriv / a.value};
}
template <typename T> GDual<T> sin(const GDual<T>& a) {
    return {std::sin(a.value), a.deriv * std::cos(a.value)};
}
template <typename T> GDual<T> cos(const GDual<T>& a) {
    return {std::cos(a.value), -a.deriv * std::sin(a.value)};
}
template <typename T> GDual<T> sqrt(const GDual<T>& a) {
    const T s = std::sqrt(a.value > T(0) ? a.value : T(0));
    return {s, s > T(0) ? a.deriv / (T(2) * s) : T(0)};
}
template <typename T> GDual<T> tanh(const GDual<T>& a) {
    const T t = std::tanh(a.value);
    return {t, a.deriv * (T(1) - t * t)};
}
template <typename T> GDual<T> abs(const GDual<T>& a) {
    return {std::abs(a.value),
            a.value > T(0) ? a.deriv : a.value < T(0) ? -a.deriv : T(0)};
}
template <typename T> GDual<T> square(const GDual<T>& a) {
    return {a.value * a.value, T(2) * a.value * a.deriv};
}
template <typename T> GDual<T> recip(const GDual<T>& a) {
    const T r = T(1) / a.value;
    return {r, -a.deriv * r * r};
}
template <typename T> GDual<T> pow(const GDual<T>& base, const GDual<T>& e) {
    const T x = base.value, y = e.value;
    T p = T(0);
    bool std_branch = false;
    if (x > T(0)) { p = std::exp(y * std::log(x)); std_branch = true; }
    else if (x == T(0) && y > T(0)) { p = T(0); }
    T dp = T(0);
    if (std_branch) {
        const T dpdx = y * std::exp((y - T(1)) * std::log(x));
        const T dpdy = p * std::log(x);
        dp = dpdx * base.deriv + dpdy * e.deriv;
    }
    return {p, dp};
}

}  // namespace rsymbolic

using namespace rsymbolic;

namespace {

// ---------------------------------------------------------------------------------
// Problem: a tree + a dataset (row-major X, target y) + the true constants used to
// generate y. The fit recovers the constants from a perturbed start.
// ---------------------------------------------------------------------------------
struct Problem {
    std::string name;
    Tree tree;
    int k;                       // number of constants
    std::vector<double> truth;   // true constants
    std::vector<std::vector<double>> X;  // m rows, nf features
    std::vector<double> y;       // m targets
};

// Evaluate the model in double for ground-truth target generation.
double model_double(const Tree& t, const double* row, const std::vector<double>& c,
                    std::vector<double>& stack) {
    return evaluate<double>(t, row, c.data(), stack);
}

Problem make_problem(const std::string& name, Tree tree, std::vector<double> truth,
                     std::size_t m, int nf, std::mt19937_64& rng) {
    Problem pr;
    pr.name = name;
    pr.tree = std::move(tree);
    pr.k = static_cast<int>(truth.size());
    pr.truth = truth;
    std::uniform_real_distribution<double> u(1.0, 5.0);  // positive: safe log/sqrt/div
    pr.X.assign(m, std::vector<double>(nf));
    pr.y.resize(m);
    std::vector<double> stack;
    for (std::size_t i = 0; i < m; ++i) {
        for (int j = 0; j < nf; ++j) pr.X[i][j] = u(rng);
        pr.y[i] = model_double(pr.tree, pr.X[i].data(), truth, stack);
    }
    return pr;
}

// Trees (postfix), mirroring bench_soa_eval.cpp, ordered by transcendental density.
Tree t_poly() {  // c0*x0^2 + c1*x1 + c2
    return {constant_node(0, 0.0), variable_node(0), unary_node(UnaryOp::Square),
            binary_node(BinaryOp::Mul),
            constant_node(1, 0.0), variable_node(1), binary_node(BinaryOp::Mul),
            binary_node(BinaryOp::Add),
            constant_node(2, 0.0), binary_node(BinaryOp::Add)};
}
Tree t_rel_mass() {  // c0 / sqrt(c1 + (x0/x1)^2)  -- Add keeps the sqrt argument positive
    return {constant_node(0, 0.0),
            constant_node(1, 0.0), variable_node(0), variable_node(1),
            binary_node(BinaryOp::Div), unary_node(UnaryOp::Square),
            binary_node(BinaryOp::Add),
            unary_node(UnaryOp::Sqrt),
            binary_node(BinaryOp::Div)};
}
Tree t_trig() {  // sin(c0*x0) * cos(c1*x1) + c2
    return {constant_node(0, 0.0), variable_node(0), binary_node(BinaryOp::Mul),
            unary_node(UnaryOp::Sin),
            constant_node(1, 0.0), variable_node(1), binary_node(BinaryOp::Mul),
            unary_node(UnaryOp::Cos),
            binary_node(BinaryOp::Mul),
            constant_node(2, 0.0), binary_node(BinaryOp::Add)};
}
Tree t_transc() {  // exp(c0*x0) + log(c1 + x1^2) + sin(c2*x0)
    return {constant_node(0, 0.0), variable_node(0), binary_node(BinaryOp::Mul),
            unary_node(UnaryOp::Exp),
            constant_node(1, 0.0), variable_node(1), unary_node(UnaryOp::Square),
            binary_node(BinaryOp::Add), unary_node(UnaryOp::Log),
            binary_node(BinaryOp::Add),
            constant_node(2, 0.0), variable_node(0), binary_node(BinaryOp::Mul),
            unary_node(UnaryOp::Sin), binary_node(BinaryOp::Add)};
}

// ---------------------------------------------------------------------------------
// Residual + Jacobian in precision T, via the production evaluate<T> / evaluate<GDual<T>>.
//   residual:  r_i = model(X_i; c) - y_i
//   jacobian:  J_ij = d r_i / d c_j   (k forward-mode dual passes, one per constant)
// ---------------------------------------------------------------------------------
template <typename T>
void residuals_T(const Problem& pr, const std::vector<T>& c, std::vector<T>& r,
                 std::vector<T>& stack) {
    const std::size_t m = pr.y.size();
    for (std::size_t i = 0; i < m; ++i)
        r[i] = evaluate<T>(pr.tree, pr.X[i].data(), c.data(), stack) -
               static_cast<T>(pr.y[i]);
}

template <typename T>
void jacobian_T(const Problem& pr, const std::vector<T>& c, std::vector<T>& J,
                std::vector<GDual<T>>& cd, std::vector<GDual<T>>& stack) {
    const std::size_t m = pr.y.size();
    const int k = pr.k;
    for (int j = 0; j < k; ++j) {
        for (int a = 0; a < k; ++a) cd[a] = GDual<T>(c[a], a == j ? T(1) : T(0));
        for (std::size_t i = 0; i < m; ++i) {
            const GDual<T> v = evaluate<GDual<T>>(pr.tree, pr.X[i].data(), cd.data(),
                                                  stack);
            J[i * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)] = v.deriv;
        }
    }
}

template <typename T>
T clamp_finite(T v) {
    return std::isfinite(v) ? v : static_cast<T>(1.0e10);
}

template <typename T>
T sse_of(const std::vector<T>& r, std::size_t m) {
    T s = T(0);
    for (std::size_t i = 0; i < m; ++i) {
        const T v = clamp_finite(r[i]);
        s += v * v;
    }
    return s;
}

// SPD solve via in-place Cholesky (M x = b), mirroring self_lm_optimizer.cpp::solve_spd.
template <typename T>
bool solve_spd(std::vector<T>& aug, std::vector<T>& rhs, int k) {
    for (int j = 0; j < k; ++j) {
        T diag = aug[static_cast<std::size_t>(j) * k + j];
        for (int p = 0; p < j; ++p) {
            const T ljp = aug[static_cast<std::size_t>(j) * k + p];
            diag -= ljp * ljp;
        }
        if (!(diag > T(0))) return false;
        const T ljj = std::sqrt(diag);
        aug[static_cast<std::size_t>(j) * k + j] = ljj;
        for (int i = j + 1; i < k; ++i) {
            T s = aug[static_cast<std::size_t>(i) * k + j];
            for (int p = 0; p < j; ++p)
                s -= aug[static_cast<std::size_t>(i) * k + p] *
                     aug[static_cast<std::size_t>(j) * k + p];
            aug[static_cast<std::size_t>(i) * k + j] = s / ljj;
        }
    }
    for (int i = 0; i < k; ++i) {
        T s = rhs[static_cast<std::size_t>(i)];
        for (int p = 0; p < i; ++p)
            s -= aug[static_cast<std::size_t>(i) * k + p] * rhs[static_cast<std::size_t>(p)];
        rhs[static_cast<std::size_t>(i)] = s / aug[static_cast<std::size_t>(i) * k + i];
    }
    for (int i = k - 1; i >= 0; --i) {
        T s = rhs[static_cast<std::size_t>(i)];
        for (int p = i + 1; p < k; ++p)
            s -= aug[static_cast<std::size_t>(p) * k + i] * rhs[static_cast<std::size_t>(p)];
        rhs[static_cast<std::size_t>(i)] = s / aug[static_cast<std::size_t>(i) * k + i];
    }
    return true;
}

// ---------------------------------------------------------------------------------
// Levenberg-Marquardt in precision T (mirrors self_lm_optimizer.cpp's run_lm_from).
// Returns optimized constants (as double) and the SSE at them (double, recomputed).
// ---------------------------------------------------------------------------------
template <typename T>
struct Scratch {
    std::vector<T> r, rt, J, A, aug, g, delta, params, trial;
    std::vector<GDual<T>> cd, gstack;
    std::vector<T> estack;
    void size(std::size_t m, int k) {
        const std::size_t ku = static_cast<std::size_t>(k);
        r.resize(m); rt.resize(m); J.resize(m * ku); A.resize(ku * ku);
        aug.resize(ku * ku); g.resize(ku); delta.resize(ku);
        params.resize(ku); trial.resize(ku); cd.resize(ku);
    }
};

template <typename T>
T run_lm(const Problem& pr, const std::vector<T>& x0, int max_iter, Scratch<T>& s) {
    const std::size_t m = pr.y.size();
    const int k = pr.k;
    const std::size_t ku = static_cast<std::size_t>(k);
    constexpr T kLambdaInit = T(1.0e-3), kLambdaUp = T(10), kLambdaDown = T(0.1);
    constexpr int kMaxInner = 30;

    s.params = x0;
    residuals_T<T>(pr, s.params, s.r, s.estack);
    T sse = sse_of<T>(s.r, m);
    T lambda = T(-1);

    for (int iter = 0; iter < max_iter; ++iter) {
        jacobian_T<T>(pr, s.params, s.J, s.cd, s.gstack);
        for (auto& e : s.J) e = clamp_finite(e);
        // A = JtJ, g = Jtr
        for (int a = 0; a < k; ++a) {
            T ga = T(0);
            for (std::size_t i = 0; i < m; ++i)
                ga += s.J[i * ku + static_cast<std::size_t>(a)] * s.r[i];
            s.g[static_cast<std::size_t>(a)] = ga;
            for (int b = a; b < k; ++b) {
                T acc = T(0);
                for (std::size_t i = 0; i < m; ++i)
                    acc += s.J[i * ku + static_cast<std::size_t>(a)] *
                           s.J[i * ku + static_cast<std::size_t>(b)];
                s.A[static_cast<std::size_t>(a) * ku + b] = acc;
                s.A[static_cast<std::size_t>(b) * ku + a] = acc;
            }
        }
        if (lambda < T(0)) {
            T md = T(0);
            for (int a = 0; a < k; ++a) md = std::max(md, s.A[static_cast<std::size_t>(a) * ku + a]);
            lambda = kLambdaInit * (md > T(0) ? md : T(1));
        }
        const T sse_before = sse;
        bool accepted = false;
        for (int inner = 0; inner < kMaxInner; ++inner) {
            for (int a = 0; a < k; ++a) {
                for (int b = 0; b < k; ++b)
                    s.aug[static_cast<std::size_t>(a) * ku + b] =
                        s.A[static_cast<std::size_t>(a) * ku + b];
                const T d = s.A[static_cast<std::size_t>(a) * ku + a];
                s.aug[static_cast<std::size_t>(a) * ku + a] += lambda * (d > T(0) ? d : T(1));
            }
            for (int a = 0; a < k; ++a) s.delta[static_cast<std::size_t>(a)] = -s.g[static_cast<std::size_t>(a)];
            if (!solve_spd<T>(s.aug, s.delta, k)) { lambda *= kLambdaUp; continue; }
            for (int a = 0; a < k; ++a)
                s.trial[static_cast<std::size_t>(a)] =
                    s.params[static_cast<std::size_t>(a)] + s.delta[static_cast<std::size_t>(a)];
            residuals_T<T>(pr, s.trial, s.rt, s.estack);
            const T sse_trial = sse_of<T>(s.rt, m);
            if (sse_trial < sse) {
                std::swap(s.params, s.trial);
                std::swap(s.r, s.rt);
                sse = sse_trial;
                lambda *= kLambdaDown;
                accepted = true;
                break;
            }
            lambda *= kLambdaUp;
        }
        if (!accepted) break;
        if (sse_before - sse <= T(1e-12) * sse_before) break;
    }
    return sse;
}

// ---------------------------------------------------------------------------------
// BFGS in precision T: minimise f(c) = 1/2 * sum r_i^2. Gradient = Jt r. Inverse-Hessian
// BFGS update with backtracking Armijo line search (Optim.jl / PySR default solver).
// ---------------------------------------------------------------------------------
template <typename T>
T run_bfgs(const Problem& pr, const std::vector<T>& x0, int max_iter, Scratch<T>& s,
           std::vector<T>& H, std::vector<T>& gprev, std::vector<T>& xprev,
           std::vector<T>& dir, std::vector<T>& xnew, std::vector<T>& gnew) {
    const std::size_t m = pr.y.size();
    const int k = pr.k;
    const std::size_t ku = static_cast<std::size_t>(k);

    auto grad_and_f = [&](const std::vector<T>& x, std::vector<T>& g) -> T {
        residuals_T<T>(pr, x, s.r, s.estack);
        const T f = T(0.5) * sse_of<T>(s.r, m);
        jacobian_T<T>(pr, x, s.J, s.cd, s.gstack);
        for (int a = 0; a < k; ++a) {
            T ga = T(0);
            for (std::size_t i = 0; i < m; ++i)
                ga += clamp_finite(s.J[i * ku + static_cast<std::size_t>(a)]) * s.r[i];
            g[static_cast<std::size_t>(a)] = ga;
        }
        return f;
    };
    auto f_only = [&](const std::vector<T>& x) -> T {
        residuals_T<T>(pr, x, s.rt, s.estack);
        return T(0.5) * sse_of<T>(s.rt, m);
    };

    s.params = x0;
    // H = identity
    std::fill(H.begin(), H.end(), T(0));
    for (int a = 0; a < k; ++a) H[static_cast<std::size_t>(a) * ku + a] = T(1);
    T f = grad_and_f(s.params, gprev);

    constexpr T c1 = T(1e-4);
    for (int iter = 0; iter < max_iter; ++iter) {
        // dir = -H g
        for (int a = 0; a < k; ++a) {
            T d = T(0);
            for (int b = 0; b < k; ++b)
                d -= H[static_cast<std::size_t>(a) * ku + b] * gprev[static_cast<std::size_t>(b)];
            dir[static_cast<std::size_t>(a)] = d;
        }
        // directional derivative g.dir (should be < 0)
        T gd = T(0);
        for (int a = 0; a < k; ++a) gd += gprev[static_cast<std::size_t>(a)] * dir[static_cast<std::size_t>(a)];
        if (!(gd < T(0))) {
            // not a descent direction (H drifted); reset to steepest descent
            for (int a = 0; a < k; ++a) dir[static_cast<std::size_t>(a)] = -gprev[static_cast<std::size_t>(a)];
            gd = T(0);
            for (int a = 0; a < k; ++a) gd -= gprev[static_cast<std::size_t>(a)] * gprev[static_cast<std::size_t>(a)];
        }
        // backtracking Armijo line search
        T alpha = T(1);
        T fnew = f;
        bool ok = false;
        for (int ls = 0; ls < 30; ++ls) {
            for (int a = 0; a < k; ++a)
                xnew[static_cast<std::size_t>(a)] =
                    s.params[static_cast<std::size_t>(a)] + alpha * dir[static_cast<std::size_t>(a)];
            fnew = f_only(xnew);
            if (std::isfinite(fnew) && fnew <= f + c1 * alpha * gd) { ok = true; break; }
            alpha *= T(0.5);
        }
        if (!ok) break;
        T gnf = grad_and_f(xnew, gnew);
        (void)gnf;
        // s_vec = xnew - params ; y_vec = gnew - gprev
        for (int a = 0; a < k; ++a) {
            xprev[static_cast<std::size_t>(a)] =
                xnew[static_cast<std::size_t>(a)] - s.params[static_cast<std::size_t>(a)];  // s
        }
        T sy = T(0);
        for (int a = 0; a < k; ++a)
            sy += xprev[static_cast<std::size_t>(a)] *
                  (gnew[static_cast<std::size_t>(a)] - gprev[static_cast<std::size_t>(a)]);
        if (sy > T(0)) {
            // BFGS inverse-Hessian update: H = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
            const T rho = T(1) / sy;
            // Hy = H * y
            static thread_local std::vector<T> Hy;
            Hy.assign(ku, T(0));
            for (int a = 0; a < k; ++a) {
                T v = T(0);
                for (int b = 0; b < k; ++b)
                    v += H[static_cast<std::size_t>(a) * ku + b] *
                         (gnew[static_cast<std::size_t>(b)] - gprev[static_cast<std::size_t>(b)]);
                Hy[static_cast<std::size_t>(a)] = v;
            }
            T yHy = T(0);
            for (int a = 0; a < k; ++a)
                yHy += (gnew[static_cast<std::size_t>(a)] - gprev[static_cast<std::size_t>(a)]) *
                       Hy[static_cast<std::size_t>(a)];
            // H += ((sy + yHy) * rho^2) s s^T - rho (Hy s^T + s (Hy)^T)
            const T coef = (sy + yHy) * rho * rho;
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b)
                    H[static_cast<std::size_t>(a) * ku + b] +=
                        coef * xprev[static_cast<std::size_t>(a)] * xprev[static_cast<std::size_t>(b)] -
                        rho * (Hy[static_cast<std::size_t>(a)] * xprev[static_cast<std::size_t>(b)] +
                               xprev[static_cast<std::size_t>(a)] * Hy[static_cast<std::size_t>(b)]);
        }
        std::swap(s.params, xnew);
        std::swap(gprev, gnew);
        f = fnew;
        // gradient convergence
        T gmax = T(0);
        for (int a = 0; a < k; ++a) gmax = std::max(gmax, std::abs(gprev[static_cast<std::size_t>(a)]));
        if (gmax <= T(1e-12)) break;
    }
    // return SSE (= 2f) recomputed
    residuals_T<T>(pr, s.params, s.r, s.estack);
    return sse_of<T>(s.r, m);
}

// Multi-start wrapper (SR.jl parity): start 0 from x0, then n_restarts perturbed starts
// xt = x0 * (1 + 0.5 * N(0,1)); keep the best by SSE. Same for both optimizers.
enum class Method { LM, BFGS };

template <typename T>
double fit_multistart(const Problem& pr, const std::vector<double>& x0d, Method method,
                      int n_restarts, int max_iter, std::uint64_t seed,
                      std::vector<double>& best_consts) {
    const int k = pr.k;
    const std::size_t m = pr.y.size();
    Scratch<T> s;
    s.size(m, k);
    s.gstack.reserve(pr.tree.size());
    s.estack.reserve(pr.tree.size());
    const std::size_t ku = static_cast<std::size_t>(k);
    std::vector<T> H(ku * ku), gprev(ku), xprev(ku), dir(ku), xnew(ku), gnew(ku);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);

    std::vector<T> x0(ku);
    for (int a = 0; a < k; ++a) x0[static_cast<std::size_t>(a)] = static_cast<T>(x0d[static_cast<std::size_t>(a)]);

    double best_sse = std::numeric_limits<double>::infinity();
    auto run_one = [&](const std::vector<T>& start) {
        T sse = method == Method::LM
                    ? run_lm<T>(pr, start, max_iter, s)
                    : run_bfgs<T>(pr, start, max_iter, s, H, gprev, xprev, dir, xnew, gnew);
        // recompute SSE in double for an unbiased quality metric
        std::vector<double> cd(ku);
        for (int a = 0; a < k; ++a) cd[static_cast<std::size_t>(a)] = static_cast<double>(s.params[static_cast<std::size_t>(a)]);
        std::vector<double> rd(m);
        std::vector<double> dstack;
        double dsse = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double v = evaluate<double>(pr.tree, pr.X[i].data(), cd.data(), dstack) - pr.y[i];
            dsse += v * v;
        }
        (void)sse;
        if (dsse < best_sse) { best_sse = dsse; best_consts = cd; }
    };

    run_one(x0);
    std::vector<T> xt(ku);
    for (int r = 0; r < n_restarts; ++r) {
        for (int a = 0; a < k; ++a)
            xt[static_cast<std::size_t>(a)] = x0[static_cast<std::size_t>(a)] *
                static_cast<T>(1.0 + 0.5 * g(rng));
        run_one(xt);
    }
    return best_sse;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double max_rel_err(const std::vector<double>& a, const std::vector<double>& truth) {
    double e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - truth[i]) / (std::abs(truth[i]) + 1e-12));
    return e;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::size_t m = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000;
    const int reps      = argc > 2 ? std::atoi(argv[2]) : 400;
    const int n_restarts = 2;   // PySR optimizer_nrestarts=2
    const int max_iter   = 8;   // PySR optimizer_iterations=8

    std::mt19937_64 dgen(12345);
    std::vector<Problem> problems = {
        make_problem("poly (arith,k=3)",    t_poly(),     {0.5, 1.3, 0.7},   m, 2, dgen),
        make_problem("rel_mass (sqrt,k=2)",  t_rel_mass(), {2.0, 1.0},        m, 2, dgen),
        make_problem("trig (sin*cos,k=3)",   t_trig(),     {1.1, 0.9, 0.3},   m, 2, dgen),
        make_problem("transc (exp/log,k=3)", t_transc(),   {0.4, 1.3, 0.8},   m, 2, dgen),
    };

    std::printf("bench_opt_precision  m=%zu  reps=%d  n_restarts=%d  max_iter=%d\n",
                m, reps, n_restarts, max_iter);
    std::printf("(per-fit wall time; final SSE & max const rel-err are quality, lower=better)\n\n");
    std::printf("%-22s %-12s %11s %11s %12s\n",
                "problem", "method", "us/fit", "SSE", "const-relerr");

    struct Row { std::string method; double us, sse, err; };

    for (Problem& pr : problems) {
        // Common perturbed start (same across all 4 cells for fairness), seed per problem.
        std::mt19937_64 sg(777);
        std::normal_distribution<double> gg(0.0, 1.0);
        std::vector<double> x0(pr.k);
        for (int a = 0; a < pr.k; ++a) x0[a] = pr.truth[a] * (1.0 + 0.3 * gg(sg));

        struct Cell { const char* tag; Method method; bool f32; };
        std::vector<Cell> cells = {
            {"LM   f64",   Method::LM,   false},
            {"LM   f32",   Method::LM,   true},
            {"BFGS f64",   Method::BFGS, false},
            {"BFGS f32",   Method::BFGS, true},
        };

        for (const Cell& cell : cells) {
            std::vector<double> consts;
            double sse = 0.0;
            // warmup
            if (cell.f32) sse = fit_multistart<float>(pr, x0, cell.method, n_restarts, max_iter, 1, consts);
            else          sse = fit_multistart<double>(pr, x0, cell.method, n_restarts, max_iter, 1, consts);
            double err = max_rel_err(consts, pr.truth);

            std::vector<double> times;
            times.reserve(reps);
            for (int r = 0; r < reps; ++r) {
                std::vector<double> c2;
                auto t0 = std::chrono::steady_clock::now();
                if (cell.f32) fit_multistart<float>(pr, x0, cell.method, n_restarts, max_iter, 1, c2);
                else          fit_multistart<double>(pr, x0, cell.method, n_restarts, max_iter, 1, c2);
                auto t1 = std::chrono::steady_clock::now();
                times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::printf("%-22s %-12s %11.1f %11.2e %12.2e\n",
                        pr.name.c_str(), cell.tag, median(times), sse, err);
        }
        std::printf("\n");
    }
    return 0;
}
