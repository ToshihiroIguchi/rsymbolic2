// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.

#include "rsymbolic/platform/libm.hpp"

// Everything here exists only on MinGW; see rsymbolic/platform/libm.hpp for why.
// On every other toolchain this translation unit is empty by design.

#if defined(RSYMBOLIC2_UCRT_LIBM)

#include <windows.h>

namespace rsymbolic {
namespace libm {
namespace detail {

namespace {

// Fallbacks, and the constant initialiser for g_ucrt below. Taking the address of
// an overloaded std:: function is not itself a constant expression, so these thin
// wrappers exist to give the table a static (load-time) initialiser rather than a
// dynamic one — that is what makes an early caller safe rather than a null call.
double std_exp(double x) { return std::exp(x); }
double std_log(double x) { return std::log(x); }
double std_sin(double x) { return std::sin(x); }
double std_cos(double x) { return std::cos(x); }
double std_erf(double x) { return std::erf(x); }
double std_pow(double x, double y) { return std::pow(x, y); }

// GetModuleHandle only, never LoadLibrary: a DLL's dynamic initialisers run under
// the Windows loader lock, where LoadLibrary can deadlock. Both names below are
// already resolved in any process that links the UCRT — which this toolchain does
// (docs/58 §2.1) — so looking them up cannot fail in the supported configuration,
// and if it ever does the table simply keeps its std:: entries.
void* ucrt_module() {
    if (void* m = ::GetModuleHandleA("ucrtbase.dll")) return m;
    return ::GetModuleHandleA("api-ms-win-crt-math-l1-1-0.dll");
}

// Resolve the plain apiset names, not the `_o_`-prefixed ones. Both are exported
// and return bit-identical results, but the `_o_` entries reach UCRT's legacy
// implementations and measured 2.2-3.8x slower than these (docs/68 §3.1) — which
// is also why this is a runtime lookup rather than a much simpler direct link
// against the `_o_` symbols.
bool resolve() {
    void* const m = ucrt_module();
    if (m == nullptr) return false;
    const HMODULE h = static_cast<HMODULE>(m);
    auto one = [h](const char* name) {
        return reinterpret_cast<double (*)(double)>(
            reinterpret_cast<void*>(::GetProcAddress(h, name)));
    };
    auto two = [h](const char* name) {
        return reinterpret_cast<double (*)(double, double)>(
            reinterpret_cast<void*>(::GetProcAddress(h, name)));
    };
    Table t;
    t.fexp = one("exp");
    t.flog = one("log");
    t.fsin = one("sin");
    t.fcos = one("cos");
    t.ferf = one("erf");
    t.fpow = two("pow");
    // All or nothing: a partially resolved table would make the Windows build's
    // numerics depend on which symbols happened to be present, which is worse
    // than being uniformly slow.
    if (!t.fexp || !t.flog || !t.fsin || !t.fcos || !t.ferf || !t.fpow) return false;
    g_ucrt = t;
    return true;
}

// Dynamic initialiser. Runs after g_ucrt's static initialisation below and before
// any search does; the write is finished before any worker thread exists, so the
// later reads need no synchronisation.
const bool g_resolved = resolve();

}  // namespace

Table g_ucrt = {std_exp, std_log, std_sin, std_cos, std_erf, std_pow};

}  // namespace detail
}  // namespace libm
}  // namespace rsymbolic

#endif  // RSYMBOLIC2_UCRT_LIBM
