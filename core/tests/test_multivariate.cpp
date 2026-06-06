// Tests for multi-feature (num_features > 1) evaluation, AD, least-squares fitting,
// and random-tree / mutation correctness.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rsymbolic/evolution/mutation.hpp"
#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/optimizer_factory.hpp"

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
#define CHECK_NEAR(a, b, tol) \
    check(std::abs((a) - (b)) < (tol), #a " ~ " #b, __FILE__, __LINE__)

using namespace rsymbolic;

// Build the postfix tree for:  x0 * x1 + c0
//   postfix: [x0, x1, Mul, c0, Add]
Tree make_x0_mul_x1_plus_c() {
    Tree t;
    t.push_back(variable_node(0));
    t.push_back(variable_node(1));
    t.push_back(binary_node(BinaryOp::Mul));
    t.push_back(constant_node(0, 0.0));
    t.push_back(binary_node(BinaryOp::Add));
    return t;
}

}  // namespace

int main() {
    // ------------------------------------------------------------------
    // 1. Evaluation of x0*x1 + c0 at several (x0, x1) pairs.
    // ------------------------------------------------------------------
    {
        Tree tree = make_x0_mul_x1_plus_c();
        const std::vector<double> c = {1.5};  // c0 = 1.5

        const double row0[] = {2.0, 3.0};  // expected: 2*3 + 1.5 = 7.5
        const double row1[] = {-1.0, 4.0}; // expected: -1*4 + 1.5 = -2.5
        const double row2[] = {0.0, 0.0};  // expected: 0 + 1.5 = 1.5

        CHECK_NEAR(evaluate<double>(tree, row0, c.data()), 7.5,  1e-12);
        CHECK_NEAR(evaluate<double>(tree, row1, c.data()), -2.5, 1e-12);
        CHECK_NEAR(evaluate<double>(tree, row2, c.data()), 1.5,  1e-12);
    }

    // ------------------------------------------------------------------
    // 2. AD Jacobian vs finite differences for x0*x1 + c0.
    //    The tree has k=1 constant; Jacobian row i = d/dc0 (x0_i * x1_i + c0) = 1.
    // ------------------------------------------------------------------
    {
        Tree tree = make_x0_mul_x1_plus_c();
        const std::vector<std::vector<double>> X = {
            {1.0, 2.0}, {-1.0, 3.0}, {0.5, -0.5}
        };
        const std::vector<double> y = {3.5, -1.7, 2.0};  // arbitrary targets
        const std::vector<double> c0_vals = {1.5};

        const OptimizationProblem prob = make_least_squares_problem(tree, X, y, c0_vals);

        // AD Jacobian
        std::vector<double> jac_ad(X.size() * 1);
        prob.jacobian(c0_vals, jac_ad);

        // Finite-difference Jacobian
        const double h = 1e-5;
        std::vector<double> r_plus(X.size()), r_minus(X.size());
        const std::vector<double> c_plus  = {c0_vals[0] + h};
        const std::vector<double> c_minus = {c0_vals[0] - h};
        prob.residuals(c_plus,  r_plus);
        prob.residuals(c_minus, r_minus);

        for (std::size_t i = 0; i < X.size(); ++i) {
            const double fd = (r_plus[i] - r_minus[i]) / (2.0 * h);
            CHECK_NEAR(jac_ad[i], fd, 1e-5);
        }
    }

    // ------------------------------------------------------------------
    // 3. Constant fitting: recover c0 = 1.5 from exact data y = x0*x1 + 1.5.
    // ------------------------------------------------------------------
    {
        Tree tree = make_x0_mul_x1_plus_c();
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                X.push_back({static_cast<double>(i), static_cast<double>(j)});
                y.push_back(static_cast<double>(i * j) + 1.5);
            }
        }

        const std::vector<double> init = {0.0};  // start far from truth
        const OptimizationProblem prob = make_least_squares_problem(tree, X, y, init);

        const auto optimizer = OptimizerFactory::create(OptimizerType::EigenLM, {});
        const OptimizationResult res = optimizer->optimize(prob);

        CHECK(res.success);
        CHECK(res.constants.size() == 1);
        CHECK_NEAR(res.constants[0], 1.5, 1e-6);
    }

    // ------------------------------------------------------------------
    // 4. generate_random_tree with num_features=3: all variable nodes must
    //    have index in [0, 2].
    // ------------------------------------------------------------------
    {
        SearchSpace space;
        space.num_features = 3;
        space.binary_ops = {BinaryOp::Add, BinaryOp::Mul};
        space.unary_ops  = {};
        space.max_depth  = 4;
        space.max_nodes  = 30;

        std::mt19937_64 rng(42);
        for (int trial = 0; trial < 500; ++trial) {
            const Tree t = generate_random_tree(space, rng);
            CHECK(is_valid_postfix(t));
            for (const Node& n : t) {
                if (n.kind == NodeKind::Variable) {
                    CHECK(n.index >= 0 && n.index < 3);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 5. Mutations with num_features=3 keep variable nodes in [0, 2].
    // ------------------------------------------------------------------
    {
        SearchSpace space;
        space.num_features = 3;
        space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
        space.unary_ops  = {};
        space.max_depth  = 4;
        space.max_nodes  = 30;

        std::mt19937_64 rng(99);
        Tree tree = generate_random_tree(space, rng);

        for (int step = 0; step < 1000; ++step) {
            mutate(tree, space, rng, 0.5);
            CHECK(is_valid_postfix(tree));
            for (const Node& n : tree) {
                if (n.kind == NodeKind::Variable) {
                    CHECK(n.index >= 0 && n.index < 3);
                }
            }
        }
    }

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
