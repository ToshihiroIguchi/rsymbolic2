// libm cost probe across platforms (docs/67).
//
// The question this exists to answer: docs/60 §7.7 established that ~97 % of
// forward-path cost is inside scalar libm calls. That makes libm the only
// remaining lever of any size — but it does NOT say whether the cost is
// intrinsic (so only a reduced-precision approximation could help) or a
// property of one platform's libm (so a full-precision replacement helps).
//
// This driver separates the two by measuring, on the same source and the same
// machine:
//   - each std:: transcendental, per element, at the SoA tile shape (P=256);
//   - hand-written replacements at three accuracy levels, so the speed a given
//     accuracy buys is priced rather than assumed;
//   - the guarded variants, because the engine's NaN/Inf semantics are load
//     bearing (sse_current returns kInf on non-finite) and the guard is not free.
//
// The replacements here are DELIBERATELY expedient — plain Taylor/atanh series,
// not the table + minimax construction a production implementation would use.
// They exist to bound the speed available at a given accuracy, not to be shipped.
// Read the reported max relative error, not the comment next to the function:
// the degree-10 exp lands at ~3e-13 (~1350 ulp), not at 1 ulp.
//
// Build (both platforms, same flags):
//   g++ -std=c++17 -O3 -DNDEBUG standalone/benchmarks/bench_libm.cpp -o bench_libm
// or via the CMake target of the same name. NOT part of the shipped package.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

// Tile width matches the SoA evaluator's kStride so the loop shape is the one
// the engine actually runs.
constexpr int kP = 256;
constexpr int kReps = 40000;

double g_sink = 0.0;

// ------------------------------------------------------------------ exp
// exp(x) = 2^n * exp(f), f = x - n*ln2, |f| <= ln2/2. The two-part ln2 keeps the
// argument reduction exact; the polynomial degree is what sets the accuracy.
double pow2i(int n) {
    const std::uint64_t bits = static_cast<std::uint64_t>(n + 1023) << 52;
    double d;
    std::memcpy(&d, &bits, 8);
    return d;
}

constexpr double kLog2e = 1.4426950408889634;
constexpr double kLn2Hi = 6.93147180369123816490e-01;
constexpr double kLn2Lo = 1.90821492927058770002e-10;

// Two variants rather than one helper with a flag: a runtime branch inside the
// reduction defeats inlining and measurably slows the polynomial that follows,
// which would be an artefact of the harness rather than of the method.
template <bool TwoPart>
int reduce(double x, double& f) {
    const double t = x * kLog2e;
    const int n = static_cast<int>(t + (t >= 0.0 ? 0.5 : -0.5));
    f = TwoPart ? (x - n * kLn2Hi) - n * kLn2Lo : x - n * (kLn2Hi + kLn2Lo);
    return n;
}

double exp_deg10(double x) {  // measured ~3e-13
    double f;
    const int n = reduce<true>(x, f);
    double p = 1.0 / 3628800;
    p = 1.0 / 362880 + f * p;
    p = 1.0 / 40320 + f * p;
    p = 1.0 / 5040 + f * p;
    p = 1.0 / 720 + f * p;
    p = 1.0 / 120 + f * p;
    p = 1.0 / 24 + f * p;
    p = 1.0 / 6 + f * p;
    p = 0.5 + f * p;
    p = 1.0 + f * p;
    p = 1.0 + f * p;
    return p * pow2i(n);
}

double exp_deg6(double x) {  // measured ~1.6e-07
    double f;
    const int n = reduce<true>(x, f);
    const double p =
        1.0 + f * (1.0 + f * (0.5 + f * (1.0 / 6 + f * (1.0 / 24 + f * (1.0 / 120 + f * (1.0 / 720))))));
    return p * pow2i(n);
}

double exp_deg4(double x) {  // measured ~5.6e-05
    double f;
    const int n = reduce<false>(x, f);
    const double p = 1.0 + f * (1.0 + f * (0.5 + f * (1.0 / 6 + f * (1.0 / 24))));
    return p * pow2i(n);
}

// What the engine would actually need: overflow, underflow and NaN preserved,
// because the search reads non-finite losses as control flow.
double exp_deg10_guarded(double x) {
    if (!(x > -708.0 && x < 709.0)) {
        if (std::isnan(x)) return x;
        return x > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
    }
    return exp_deg10(x);
}

// ------------------------------------------------------------------ log
// log(x) = e*ln2 + 2*atanh(s), s = (m-1)/(m+1), m in [sqrt(0.5), sqrt(2)).
double log_split(double x, int& e) {
    std::uint64_t b;
    std::memcpy(&b, &x, 8);
    e = static_cast<int>((b >> 52) & 0x7ff) - 1023;
    b = (b & 0x800FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m;
    std::memcpy(&m, &b, 8);
    if (m > 1.4142135623730951) {
        m *= 0.5;
        ++e;
    }
    return (m - 1.0) / (m + 1.0);
}

double log_s17(double x) {  // measured ~2.2e-16 (1 ulp)
    int e;
    const double s = log_split(x, e);
    const double s2 = s * s;
    double p = 2.0 / 17;
    p = 2.0 / 15 + s2 * p;
    p = 2.0 / 13 + s2 * p;
    p = 2.0 / 11 + s2 * p;
    p = 2.0 / 9 + s2 * p;
    p = 2.0 / 7 + s2 * p;
    p = 2.0 / 5 + s2 * p;
    p = 2.0 / 3 + s2 * p;
    p = 2.0 + s2 * p;
    return (e * kLn2Hi + s * p) + e * kLn2Lo;
}

double log_s11(double x) {  // measured ~1.9e-12
    int e;
    const double s = log_split(x, e);
    const double s2 = s * s;
    const double p =
        s * (2.0 + s2 * (2.0 / 3 + s2 * (2.0 / 5 + s2 * (2.0 / 7 + s2 * (2.0 / 9 + s2 * (2.0 / 11))))));
    return e * (kLn2Hi + kLn2Lo) + p;
}

double log_s5(double x) {  // measured ~1.6e-07
    int e;
    const double s = log_split(x, e);
    const double s2 = s * s;
    const double p = s * (2.0 + s2 * (2.0 / 3 + s2 * (2.0 / 5)));
    return e * (kLn2Hi + kLn2Lo) + p;
}

double log_s17_guarded(double x) {
    if (!(x > 0.0)) {
        if (std::isnan(x)) return x;
        return x == 0.0 ? -std::numeric_limits<double>::infinity()
                        : std::numeric_limits<double>::quiet_NaN();
    }
    if (x == std::numeric_limits<double>::infinity()) return x;
    return log_s17(x);
}

double pow_acc(double a, double b) { return exp_deg10(b * log_s17(a)); }
double pow_fast(double a, double b) { return exp_deg6(b * log_s11(a)); }

// ------------------------------------------------------------------ harness
template <class F>
double time_kernel(const std::vector<double>& a, const std::vector<double>& b, F f) {
    std::vector<double> dst(kP);
    for (int r = 0; r < 200; ++r) {  // warm-up
        for (int i = 0; i < kP; ++i) dst[i] = f(a[i], b[i]);
    }
    g_sink += dst[7];
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kReps; ++r) {
        for (int i = 0; i < kP; ++i) dst[i] = f(a[i], b[i]);
        g_sink += dst[r & (kP - 1)];
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() /
           (static_cast<double>(kReps) * kP);
}

struct Row {
    const char* name;
    double ns;
};

}  // namespace

int main() {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> exponent(-5.0, 5.0);
    std::uniform_real_distribution<double> positive(0.05, 50.0);
    std::vector<double> a(kP), b(kP);
    for (int i = 0; i < kP; ++i) {
        a[i] = positive(rng);
        b[i] = exponent(rng);
    }

    std::printf("bench_libm  tile P=%d  reps=%d\n\n", kP, kReps);

    std::vector<Row> rows;
    rows.push_back({"copy (baseline)", time_kernel(a, b, [](double x, double) { return x; })});
    rows.push_back({"mul", time_kernel(a, b, [](double x, double y) { return x * y; })});
    rows.push_back({"std::exp", time_kernel(a, b, [](double, double y) { return std::exp(y); })});
    rows.push_back({"  exp deg10", time_kernel(a, b, [](double, double y) { return exp_deg10(y); })});
    rows.push_back({"  exp deg10 guarded", time_kernel(a, b, [](double, double y) { return exp_deg10_guarded(y); })});
    rows.push_back({"  exp deg6", time_kernel(a, b, [](double, double y) { return exp_deg6(y); })});
    rows.push_back({"  exp deg4", time_kernel(a, b, [](double, double y) { return exp_deg4(y); })});
    rows.push_back({"std::log", time_kernel(a, b, [](double x, double) { return std::log(x); })});
    rows.push_back({"  log s17", time_kernel(a, b, [](double x, double) { return log_s17(x); })});
    rows.push_back({"  log s17 guarded", time_kernel(a, b, [](double x, double) { return log_s17_guarded(x); })});
    rows.push_back({"  log s11", time_kernel(a, b, [](double x, double) { return log_s11(x); })});
    rows.push_back({"  log s5", time_kernel(a, b, [](double x, double) { return log_s5(x); })});
    rows.push_back({"std::sin", time_kernel(a, b, [](double, double y) { return std::sin(y); })});
    rows.push_back({"std::sqrt", time_kernel(a, b, [](double x, double) { return std::sqrt(x); })});
    rows.push_back({"std::pow", time_kernel(a, b, [](double x, double y) { return std::pow(x, y); })});
    rows.push_back({"  pow acc (exp.log)", time_kernel(a, b, [](double x, double y) { return pow_acc(x, y); })});
    rows.push_back({"  pow fast (exp.log)", time_kernel(a, b, [](double x, double y) { return pow_fast(x, y); })});

    const double base = rows[0].ns;
    std::printf("%-24s %10s %12s\n", "kernel", "ns/elem", "net(-copy)");
    for (const Row& r : rows) std::printf("%-24s %10.3f %12.3f\n", r.name, r.ns, r.ns - base);

    // Accuracy is measured, not asserted: the speed of an approximation is
    // meaningless without the error it was bought with.
    std::mt19937_64 r2(777);
    std::uniform_real_distribution<double> dx(-30.0, 30.0);
    std::uniform_real_distribution<double> dl(1e-8, 1e8);
    double e10 = 0, e6 = 0, e4 = 0, l17 = 0, l11 = 0, l5 = 0, pa = 0, pf = 0;
    for (int i = 0; i < 2000000; ++i) {
        const double x = dx(r2);
        const double ref = std::exp(x);
        e10 = std::max(e10, std::fabs(exp_deg10(x) - ref) / ref);
        e6 = std::max(e6, std::fabs(exp_deg6(x) - ref) / ref);
        e4 = std::max(e4, std::fabs(exp_deg4(x) - ref) / ref);
        const double v = dl(r2);
        const double rl = std::log(v);
        if (std::fabs(rl) > 1e-3) {
            l17 = std::max(l17, std::fabs(log_s17(v) - rl) / std::fabs(rl));
            l11 = std::max(l11, std::fabs(log_s11(v) - rl) / std::fabs(rl));
            l5 = std::max(l5, std::fabs(log_s5(v) - rl) / std::fabs(rl));
        }
        const double e = dx(r2) * 0.1;
        const double rp = std::pow(v, e);
        if (std::isfinite(rp) && rp != 0.0) {
            pa = std::max(pa, std::fabs(pow_acc(v, e) - rp) / std::fabs(rp));
            pf = std::max(pf, std::fabs(pow_fast(v, e) - rp) / std::fabs(rp));
        }
    }
    std::printf("\nmax relative error vs libm, 2e6 samples (1 ulp = 2.2e-16)\n");
    std::printf("  exp deg10/deg6/deg4 : %.3e  %.3e  %.3e\n", e10, e6, e4);
    std::printf("  log s17/s11/s5      : %.3e  %.3e  %.3e\n", l17, l11, l5);
    std::printf("  pow acc/fast        : %.3e  %.3e\n", pa, pf);

    // Keep the accumulated results observable so the timed loops cannot be
    // optimised away; the value itself is not meaningful.
    std::printf("\nsink %g\n", g_sink);
    return 0;
}
