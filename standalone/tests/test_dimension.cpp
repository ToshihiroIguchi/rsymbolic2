// Unit tests for the Dimension type (fixed-denominator rational SI-dimension vector).

#include <cstdio>

#include "rsymbolic/units/dimension.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s (%s:%d)\n", expr, file, line);
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

using rsymbolic::Dimension;
using rsymbolic::base_dim;

// Axis constants mirroring dimension.hpp's order.
enum { MASS = 0, LENGTH = 1, TIME = 2, CURRENT = 3, TEMP = 4, AMOUNT = 5, LUM = 6 };

void test_base_and_dimensionless() {
    const Dimension dimensionless;  // all-zero
    CHECK(dimensionless.dimensionless());

    const Dimension m = base_dim(LENGTH);
    CHECK(!m.dimensionless());
    CHECK(m.num[LENGTH] == Dimension::DEN);
    CHECK(m.num[TIME] == 0);
    CHECK(m == base_dim(LENGTH));
    CHECK(m != base_dim(TIME));
}

// velocity = m / s ; force = kg * m / s^2 ; N should equal kg*m*s^-2.
void test_mul_div() {
    const Dimension m = base_dim(LENGTH);
    const Dimension s = base_dim(TIME);
    const Dimension kg = base_dim(MASS);

    const Dimension velocity = m - s;  // m / s
    CHECK(velocity.num[LENGTH] == Dimension::DEN);
    CHECK(velocity.num[TIME] == -Dimension::DEN);

    const Dimension accel = m - s.scaled(2);  // m / s^2
    const Dimension force = kg + accel;       // kg * (m/s^2)
    const Dimension newton = kg + m - s.scaled(2);
    CHECK(force == newton);
    CHECK(force.num[MASS] == Dimension::DEN);
    CHECK(force.num[LENGTH] == Dimension::DEN);
    CHECK(force.num[TIME] == -2 * Dimension::DEN);
}

void test_reciprocal() {
    const Dimension s = base_dim(TIME);
    const Dimension hz = -s;  // 1 / s
    CHECK(hz.num[TIME] == -Dimension::DEN);
    CHECK((s + hz).dimensionless());  // s * (1/s) = dimensionless
}

// sqrt(m^2) == m exactly; square(sqrt(m)) == m; equality is exact after round trips.
void test_scaled_rooted_exact() {
    const Dimension m = base_dim(LENGTH);

    const Dimension m2 = m.scaled(2);  // m^2
    Dimension root;
    CHECK(m2.rooted(2, root));
    CHECK(root == m);

    // sqrt(m) then square gets back to m exactly (the FixedRational exactness point).
    Dimension half;
    CHECK(m.rooted(2, half));        // m^(1/2), exponent DEN/2 = 12600
    CHECK(half.num[LENGTH] == Dimension::DEN / 2);
    CHECK(half.scaled(2) == m);
}

// A root that does not divide evenly must fail (representability limit).
void test_root_not_divisible() {
    const Dimension m = base_dim(LENGTH);  // exponent = DEN = 25200
    Dimension out;
    // 25200 is not divisible by 32 (25200 / 32 is not integer) -> false.
    CHECK(!m.rooted(32, out));
    // But it is divisible by 2, 3, 4, 5, 6, 7 (DEN = 2^4*3^2*5^2*7).
    CHECK(m.rooted(2, out));
    CHECK(m.rooted(3, out));
    CHECK(m.rooted(7, out));
}

void test_pow_rational() {
    const Dimension m2 = base_dim(LENGTH).scaled(2);  // m^2
    Dimension out;
    // (m^2)^(1/2) = m
    CHECK(m2.pow_rational(1, 2, out));
    CHECK(out == base_dim(LENGTH));
    // (m^2)^(3/2) = m^3
    CHECK(m2.pow_rational(3, 2, out));
    CHECK(out == base_dim(LENGTH).scaled(3));
    // m^(1/2) is representable (DEN even); m^(1/32) is not.
    CHECK(base_dim(LENGTH).pow_rational(1, 2, out));
    CHECK(!base_dim(LENGTH).pow_rational(1, 32, out));
}

}  // namespace

int main() {
    test_base_and_dimensionless();
    test_mul_div();
    test_reciprocal();
    test_scaled_rooted_exact();
    test_root_not_divisible();
    test_pow_rational();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
