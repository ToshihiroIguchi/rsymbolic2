.libPaths(c("C:/Users/toshi/R/library", .libPaths()))
library(rsymbolic2)

cat("=== Test 1: linear y = 2.5x + 1.7 ===\n")
X <- matrix(seq(-5, 5, length.out=20), ncol=1)
y <- 2.5 * X[,1] + 1.7
res <- symbolic_regression(X, y, unary_ops=character(0),
       population_size=200L, generations=40L, seed=42L)
cat("expr:      ", res$expression, "\n")
cat("loss:      ", res$loss, "\n")
cat("complexity:", res$complexity, "\n")

cat("\n=== Test 2: exponential y = 2*exp(0.3x) ===\n")
X2 <- matrix(seq(0, 3, length.out=20), ncol=1)
y2 <- 2 * exp(0.3 * X2[,1])
res2 <- symbolic_regression(X2, y2, unary_ops="exp",
        population_size=400L, generations=60L, seed=7L)
cat("expr:      ", res2$expression, "\n")
cat("loss:      ", res2$loss, "\n")

cat("\n=== Test 3: 2-feature y = x0*x1 + x0 ===\n")
X3 <- as.matrix(expand.grid(seq(-2,2,length.out=5), seq(-2,2,length.out=5)))
y3 <- X3[,1]*X3[,2] + X3[,1]
res3 <- symbolic_regression(X3, y3, unary_ops=character(0),
        population_size=300L, generations=50L, seed=42L)
cat("expr:      ", res3$expression, "\n")
cat("loss:      ", res3$loss, "\n")
cat("Pareto rows:", nrow(res3$pareto_front), "\n")
print(res3$pareto_front)
