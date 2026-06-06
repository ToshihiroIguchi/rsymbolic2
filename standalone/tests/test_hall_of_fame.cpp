// Tests for the Hall of Fame (best-per-complexity archive and Pareto front).

#include <cstdio>
#include <vector>

#include "rsymbolic/evolution/hall_of_fame.hpp"
#include "rsymbolic/expression/tree.hpp"

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

using namespace rsymbolic;

PopMember member(double loss, int complexity) {
    PopMember m;
    m.loss = loss;
    m.complexity = complexity;
    return m;
}

void test_keeps_best_per_complexity() {
    HallOfFame hof;
    hof.update(member(10.0, 3));
    hof.update(member(5.0, 3));   // better at complexity 3
    hof.update(member(7.0, 3));   // worse, ignored
    CHECK(hof.best().loss == 5.0);
    CHECK(hof.best().complexity == 3);
}

void test_best_across_complexities() {
    HallOfFame hof;
    hof.update(member(8.0, 2));
    hof.update(member(3.0, 5));
    hof.update(member(6.0, 9));
    CHECK(hof.best().loss == 3.0);
    CHECK(hof.best().complexity == 5);
}

void test_pareto_front_is_non_dominated() {
    HallOfFame hof;
    hof.update(member(10.0, 1));  // simplest, worst loss   -> on front
    hof.update(member(4.0, 3));   // better loss, more complex -> on front
    hof.update(member(6.0, 5));   // dominated by (4.0, 3)  -> excluded
    hof.update(member(2.0, 7));   // better loss            -> on front

    const std::vector<PopMember> front = hof.pareto_front();
    CHECK(front.size() == 3);
    // Ascending complexity, strictly decreasing loss.
    CHECK(front[0].complexity == 1 && front[0].loss == 10.0);
    CHECK(front[1].complexity == 3 && front[1].loss == 4.0);
    CHECK(front[2].complexity == 7 && front[2].loss == 2.0);
}

void test_empty() {
    HallOfFame hof;
    CHECK(hof.empty());
    CHECK(hof.pareto_front().empty());
}

}  // namespace

int main() {
    test_keeps_best_per_complexity();
    test_best_across_complexities();
    test_pareto_front_is_non_dominated();
    test_empty();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
