// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.

#pragma once

#include <cmath>

// The platform's fastest correct libm, for the transcendentals the evaluator calls.
//
// On every toolchain except MinGW this header IS <cmath>: each function below is a
// one-line forward to std::, compiles away completely, and the Linux, WASM and MSVC
// builds emit byte-for-byte the code they emitted before this header existed.
//
// MinGW is the exception, and docs/68 is the measurement behind it. mingw-w64
// implements exp/log/sin/cos/erf/pow itself in libmingwex instead of forwarding them
// to the UCRT that the very same toolchain already links, and those implementations
// are 4.5-9.4x slower than UCRT's on the same machine. That one fact accounts for
// essentially all of the ~2.5x end-to-end penalty Windows pays against Linux — the
// gap is zero on pure arithmetic and grows in exact proportion to transcendental
// content (docs/67 §3.4). So on MinGW we call UCRT's implementations directly.
//
// Note what is deliberately NOT redirected: tanh, sinh, cosh, sqrt and abs. Those
// measured 1.04-1.17x (i.e. nothing) *and* bit-identical between the two libms,
// which is the tell that mingw-w64 already forwards them to the UCRT. Redirecting
// them would buy no speed and spend a bit-identity break to do it.
//
// Scope is the evaluator: tree.hpp, dual.hpp, multi_dual.hpp and soa_eval.hpp, the
// paths that run once per data point. The handful of libm calls in the search
// mechanics themselves (annealing in evolutionary_search.cpp, the score in
// hall_of_fame.cpp, the perturbation factor in mutation.cpp) stay on std::: they run
// once per event rather than once per point, so redirecting them would buy nothing.
//
// Under CLAUDE.md's PySR-parity rule this is an implementation-method change: it
// alters how a transcendental is computed, never which settings define the search.
// It does move Windows results by up to 1-2 ulp, which is a one-time, audited
// bit-identity break against pre-existing Windows builds (docs/68 §7); every
// other platform is untouched because it does not compile the redirect at all.

// The named fallback the Dependency Policy asks for, and the A/B switch every
// measurement in docs/68 §6 is made with: -DRSYMBOLIC2_NO_UCRT_LIBM reverts every
// function below to std::, restoring the pre-redirect Windows build exactly.
#if defined(__MINGW32__) && !defined(RSYMBOLIC2_NO_UCRT_LIBM)
#define RSYMBOLIC2_UCRT_LIBM 1
#endif

namespace rsymbolic {
namespace libm {

#if defined(RSYMBOLIC2_UCRT_LIBM)

namespace detail {

// Resolved once, before main(), in platform_libm.cpp. Constant-initialised to the
// std:: implementations there, so a caller that somehow runs before the dynamic
// initialiser — or a machine where the UCRT symbols cannot be resolved — gets
// today's behaviour rather than a null call.
struct Table {
    double (*fexp)(double);
    double (*flog)(double);
    double (*fsin)(double);
    double (*fcos)(double);
    double (*ferf)(double);
    double (*fpow)(double, double);
};

extern Table g_ucrt;

}  // namespace detail

inline double exp(double x) { return detail::g_ucrt.fexp(x); }
inline double log(double x) { return detail::g_ucrt.flog(x); }
inline double sin(double x) { return detail::g_ucrt.fsin(x); }
inline double cos(double x) { return detail::g_ucrt.fcos(x); }
inline double erf(double x) { return detail::g_ucrt.ferf(x); }
inline double pow(double x, double y) { return detail::g_ucrt.fpow(x, y); }

#else

inline double exp(double x) { return std::exp(x); }
inline double log(double x) { return std::log(x); }
inline double sin(double x) { return std::sin(x); }
inline double cos(double x) { return std::cos(x); }
inline double erf(double x) { return std::erf(x); }
inline double pow(double x, double y) { return std::pow(x, y); }

#endif

}  // namespace libm
}  // namespace rsymbolic
