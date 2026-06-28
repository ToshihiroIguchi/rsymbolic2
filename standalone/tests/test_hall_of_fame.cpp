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

void test_members_ascending_complexity() {
    HallOfFame hof;
    hof.update(member(6.0, 9));
    hof.update(member(8.0, 2));
    hof.update(member(3.0, 5));
    const std::vector<PopMember> ms = hof.members();
    CHECK(ms.size() == 3);
    CHECK(ms[0].complexity == 2);
    CHECK(ms[1].complexity == 5);
    CHECK(ms[2].complexity == 9);
}

void test_select_best_picks_knee() {
    // A Pareto front where complexity 5 gives a huge loss drop and further complexity
    // barely helps: the score (loss-drop per unit complexity, log scale) peaks at 5.
    std::vector<PopMember> front;
    front.push_back(member(100.0, 1));
    front.push_back(member(1.0, 5));     // big improvement -> highest score
    front.push_back(member(0.99, 9));    // marginal improvement -> low score
    const std::size_t idx = select_best(front);
    CHECK(idx == 1);
}

void test_select_best_edge_cases() {
    CHECK(select_best({}) == 0);                       // empty -> 0 (defined fallback)
    CHECK(select_best({member(3.0, 4)}) == 0);         // single -> that one
}

void test_model_selection_modes() {
    // A front whose largest score is the early big drop (idx 1), but whose only member
    // within 1.5x of the most accurate loss (0.001) is the last (idx 2). This separates
    // all three modes.
    std::vector<PopMember> front;
    front.push_back(member(100.0, 1));
    front.push_back(member(0.01, 3));    // biggest log-loss drop per complexity -> top score
    front.push_back(member(0.001, 9));   // most accurate; only one in the 1.5x band

    // accuracy: the lowest-loss (last) member.
    CHECK(select_best(front, ModelSelection::Accuracy) == 2);
    // score: highest score over the whole front (the early drop).
    CHECK(select_best(front, ModelSelection::Score) == 1);
    // best: highest score among members within 1.5x of min loss -> only idx 2 qualifies.
    CHECK(select_best(front, ModelSelection::Best) == 2);
    // default mode is Best.
    CHECK(select_best(front) == select_best(front, ModelSelection::Best));
}

}  // namespace

int main() {
    test_keeps_best_per_complexity();
    test_best_across_complexities();
    test_pareto_front_is_non_dominated();
    test_empty();
    test_members_ascending_complexity();
    test_select_best_picks_knee();
    test_select_best_edge_cases();
    test_model_selection_modes();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
