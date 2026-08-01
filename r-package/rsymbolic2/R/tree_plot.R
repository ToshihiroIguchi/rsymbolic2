# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.

# Equation tree diagram (docs/48 D6), the R half of a feature shared with the Python
# package and the web GUI.
#
# The tree is not exported from the C++ core: R's own parser already turns the printed
# infix string into a language object, exactly as predict.rsymbolic2() relies on. This file
# only normalises that language object into a flat node table and draws it, so nothing here
# touches the search or the stored expression strings.

# Unary operators the core can print in call form. `neg`, `square` and `inv` are no longer
# among them — the core renders those three as `(-a)`, `(a ^ 2)` and `(1 / a)` (docs/71),
# which arrive here as ordinary operators — but they stay in this list so a string saved by
# an earlier version still draws. They are not R functions (predict.rsymbolic2() supplies
# them in its evaluation environment); the rest are, but here they are only labels.
UNARY_OPS <- c("neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square", "inv",
               "erf", "sinh", "cosh")

# Binary glyphs. "/" and "*" are rewritten: "/" reads as part of a fraction that is not
# there, and "*" is a raised asterisk that sits off-centre inside a node. The web GUI and
# the Python package use the same two substitutions. Written as \u escapes so the package
# sources stay ASCII (R CMD check flags non-ASCII characters in R code).
# intToUtf8 keeps this file ASCII (R CMD check flags non-ASCII characters in R code)
# while the labels themselves stay the real multiplication and division signs.
TIMES_GLYPH  <- intToUtf8(0x00D7)
DIVIDE_GLYPH <- intToUtf8(0x00F7)
BINARY_LABEL <- c("+" = "+", "-" = "-", "*" = TIMES_GLYPH, "/" = DIVIDE_GLYPH,
                  "^" = "^")

# Constants are relabelled with the same "%.6g" printf the core's to_string() uses, so a
# node reads exactly as it does inside the expression string.
tree_constant_label <- function(v) {
    if (is.nan(v)) return("nan")
    if (is.infinite(v)) return(if (v < 0) "-inf" else "inf")
    # sprintf(), not formatC(): this is the same printf the core's to_string() calls, and
    # formatC() pads the result to a common width ("    2.2").
    sprintf("%.6g", v)
}

tree_variable_label <- function(name, variable_names) {
    idx <- suppressWarnings(as.integer(sub("^x", "", name)))
    if (!is.na(idx) && length(variable_names) > idx) {
        nm <- variable_names[[idx + 1L]]
        if (!is.na(nm) && nzchar(nm)) return(nm)
    }
    name
}

# One language object -> the node shape all three surfaces draw:
# list(kind = "operator"/"variable"/"constant", label, children).
#
# Two normalisations matter here. R's parser KEEPS parentheses as calls to `(`, unlike
# Python's ast and the browser's parser, so they are stripped or the R tree would be
# visibly larger than the other two for the same equation. And a negated literal collapses
# into one constant: "%.6g" emits "-1.3", R reads that as unary minus over 1.3, and two
# nodes there are noise (parse_expression.hpp folds the same case).
tree_draw_node <- function(e, variable_names) {
    while (is.call(e) && identical(e[[1L]], quote(`(`))) e <- e[[2L]]

    if (is.numeric(e)) {
        return(list(kind = "constant", label = tree_constant_label(e), children = list()))
    }
    if (is.name(e)) {
        name <- as.character(e)
        # `inf` / `nan` parse as names, not numbers; they are values, not data columns.
        if (name %in% c("inf", "nan", "Inf", "NaN")) {
            return(list(kind = "constant", label = tolower(name), children = list()))
        }
        return(list(kind = "variable",
                    label = tree_variable_label(name, variable_names),
                    children = list()))
    }
    if (!is.call(e)) {
        stop("cannot draw this expression: unsupported element '",
             paste(deparse(e), collapse = " "), "'.")
    }

    op <- as.character(e[[1L]])
    args <- as.list(e)[-1L]

    if (op == "-" && length(args) == 1L) {
        inner <- args[[1L]]
        while (is.call(inner) && identical(inner[[1L]], quote(`(`))) inner <- inner[[2L]]
        if (is.numeric(inner)) {
            return(list(kind = "constant", label = tree_constant_label(-inner),
                        children = list()))
        }
        # Labelled with the sign the expression string shows, not the engine's "neg":
        # one operator must not appear under two names on one screen. A one-child "-"
        # is unambiguous beside the two-child subtraction.
        return(list(kind = "operator", label = "-",
                    children = list(tree_draw_node(inner, variable_names))))
    }
    if (length(args) == 2L && op %in% names(BINARY_LABEL)) {
        return(list(kind = "operator", label = unname(BINARY_LABEL[[op]]),
                    children = lapply(args, tree_draw_node, variable_names)))
    }
    if (length(args) == 1L && op %in% UNARY_OPS) {
        return(list(kind = "operator", label = op,
                    children = list(tree_draw_node(args[[1L]], variable_names))))
    }
    stop("cannot draw this expression: unsupported operator '", op, "' with ",
         length(args), " argument(s).")
}

# Flat node table for one expression string. Internal, but the layout is the part worth
# testing directly, so the tests call it as rsymbolic2:::expr_tree_layout().
#
# Leaves take consecutive integer columns and an inner node sits at the mean of its
# children, so sibling subtrees own disjoint leaf ranges and same-depth nodes can never
# collide; a unary node lands directly above its only child. Reingold-Tilford would pack
# wide trees tighter, but at the default max_nodes = 30 this draws the same picture for a
# fraction of the code.
expr_tree_layout <- function(expr, variable_names = NULL) {
    expr <- as.character(expr)[1L]
    lang <- tryCatch(str2lang(expr), error = function(e)
        stop("could not parse the expression '", expr, "': ", conditionMessage(e)))
    root <- tree_draw_node(lang, variable_names)

    ids <- integer(0); parents <- integer(0); depths <- integer(0)
    xs <- numeric(0); labels <- character(0); kinds <- character(0)
    next_leaf <- 0

    walk <- function(node, depth, parent) {
        id <- length(ids) + 1L
        ids[id] <<- id
        parents[id] <<- parent
        depths[id] <<- depth
        labels[id] <<- node$label
        kinds[id] <<- node$kind
        xs[id] <<- NA_real_
        if (length(node$children) == 0L) {
            xs[id] <<- next_leaf
            next_leaf <<- next_leaf + 1
        } else {
            child_x <- vapply(node$children, walk, numeric(1), depth + 1L, id)
            xs[id] <<- mean(child_x)
        }
        xs[id]
    }
    walk(root, 0L, NA_integer_)

    data.frame(id = ids, parent = parents, depth = depths, x = xs,
               label = labels, kind = kinds, stringsAsFactors = FALSE)
}

# Node fills by kind: operator, variable (a data column) and constant (a fitted number).
# Uniform shape, three fills -- the distinction a symbolic-regression reader wants, and the
# same encoding the web GUI uses.
TREE_FILL <- c(operator = "#eaeef4", variable = "#e8f0ff", constant = "#f4f5f7")
TREE_COLOUR <- c(operator = "#1c2430", variable = "#1d4ed8", constant = "#5b6472")

# One expression as a tree. Nodes are capsules: a circle for the one-character operators,
# stretching for a long constant, which is what Graphviz's default `shape=ellipse` does and
# what a tree diagram conventionally shows. ggplot2 has no ellipse geom without ggforce (a
# new dependency), so the capsule is built from geom_label():
#
#   * `label.r` is the corner radius, and grid does NOT clamp it -- a radius wider than half
#     the box makes the corner arcs overlap and the text spills out of the shape. Half the
#     box height (1 line of text + 2 * 0.25 line of padding) is 0.75 lines.
#   * `label.padding` applies to all four sides, so a one-character label would be taller
#     than it is wide and the radius above would degenerate. One space on each side of the
#     label buys the width instead. It is cosmetic only: the layout table keeps the bare
#     labels, which is what the tests assert on.
#
# Box and text are both in absolute units, so the picture survives any panel resize.
tree_plot <- function(expr, variable_names = NULL, title = "rsymbolic2 equation tree") {
    nodes <- expr_tree_layout(expr, variable_names)
    nodes$y <- -nodes$depth
    nodes$drawn <- paste0(" ", nodes$label, " ")

    edges <- nodes[!is.na(nodes$parent), , drop = FALSE]
    if (nrow(edges) > 0L) {
        p_row <- match(edges$parent, nodes$id)
        edges$xend <- nodes$x[p_row]
        edges$yend <- nodes$y[p_row]
    }

    p <- ggplot2::ggplot(nodes, ggplot2::aes(x = x, y = y))
    if (nrow(edges) > 0L) {
        p <- p + ggplot2::geom_segment(
            data = edges,
            ggplot2::aes(x = x, y = y, xend = xend, yend = yend),
            colour = "grey65", inherit.aes = FALSE)
    }
    p +
        ggplot2::geom_label(
            ggplot2::aes(label = drawn, fill = kind, colour = kind),
            label.r = ggplot2::unit(0.75, "lines"),
            label.padding = ggplot2::unit(0.25, "lines"),
            size = 3.4, show.legend = FALSE) +
        ggplot2::scale_fill_manual(values = TREE_FILL) +
        ggplot2::scale_colour_manual(values = TREE_COLOUR) +
        ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = 0.08)) +
        ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = 0.12)) +
        ggplot2::labs(title = title) +
        ggplot2::theme_void() +
        ggplot2::theme(legend.position = "none",
                       plot.title = ggplot2::element_text(size = 11))
}
