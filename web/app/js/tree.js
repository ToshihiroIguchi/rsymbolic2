// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Equation tree diagram (docs/48 D6). Draws the selected equation as a syntax tree —
// operators as inner nodes, variables and constants as leaves — beside the LaTeX formula
// the hero card already renders.
//
// The tree is NOT exported from the C++ core: `parseExpression` (predict.js) already turns
// the printed infix string into an AST, exactly as the R and Python packages re-parse it in
// their own languages. This module only normalises that AST into one flat node list and
// renders it as inline SVG (no plotting library: Chart.js cannot draw trees).
//
// Node count caveat: this counts the parse of the *printed* (display-simplified, docs/52)
// expression, which is usually smaller than the `complexity` column — that counts the raw tree
// the search archived. Same relationship main.js::fmtComplexity renders as "10 → 7". It can also
// be LARGER than the simplified count, and routinely is: since docs/71 the engine prints the
// one-child nodes in notation rather than by name, so a Square the core counts as two nodes
// (square, x0) prints as "(x0 ^ 2)" and re-parses here as three. The tree draws what is printed,
// so it counts what is printed; the column's own tooltip states the same both-ways caveat.

import { parseExpression } from "./predict.js";

// Binary operator glyphs. Division and multiplication are rewritten: "/" reads as part of a
// fraction that is not there, and "*" is a raised asterisk in most fonts, which sits visibly
// off-centre inside a capsule. "÷" and "×" are centred glyphs and are what a tree diagram
// conventionally shows.
//
// Must stay character-identical to main.js's BINARY_GLYPH, which labels the operator pills
// with the same glyphs but keys them by engine id ("mul") rather than by the parsed symbol
// used here ("*"). Showing one operator under two names on one screen is exactly what that
// pair of tables exists to prevent, so change them together.
const BINARY_LABEL = { "+": "+", "-": "-", "*": "×", "/": "÷", "^": "^" };

// Geometry, in SVG user units (= CSS px at scale 1).
const NODE_H = 28; // capsule height; also the minimum width, so a 1-character label is a circle
const PAD_X = 11; // horizontal padding around the label inside the capsule
const GAP = 16; // clear space between two adjacent capsules
const LEVEL_H = 64; // vertical distance between two depths
const MARGIN = 14;
const FONT_SIZE = 13;
const FALLBACK_CHAR_W = 7.6; // used only when the SVG is not laid out yet (measurement = 0)

// --- Normalisation ----------------------------------------------------------------

// Mirrors the C++ "%.6g" of tree.hpp::to_string closely enough that a constant reads the
// same in the tree as in the equation string: 6 significant digits, exponential form
// outside [1e-4, 1e6), and the non-finite values a degenerate fit can carry.
function constLabel(v) {
  if (Number.isNaN(v)) return "nan";
  if (!Number.isFinite(v)) return v < 0 ? "-inf" : "inf";
  if (v === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(v)));
  if (e < -4 || e >= 6) {
    // Trim the mantissa's trailing zeros and pad the exponent to two digits, so the label
    // matches the "%.6g" text in the equation string above ("3.82229e-08", not "3.82229e-8").
    const [m, ex] = v.toExponential(5).replace(/\.?0+e/, "e").split("e");
    const sign = ex[0] === "-" ? "-" : "+";
    return `${m}e${sign}${ex.replace(/^[+-]/, "").padStart(2, "0")}`;
  }
  return String(Number(v.toPrecision(6)));
}

function varLabel(index, featureNames) {
  const name = featureNames && featureNames[index];
  return name != null && name !== "" ? String(name) : `x${index}`;
}

// predict.js AST -> the one node shape all three surfaces draw:
// { kind: "operator" | "variable" | "constant", label, children }.
//
// A negated literal collapses into a single constant: "%.6g" emits "-1.3", every parser
// reads that as unary minus over 1.3, and drawing it as two nodes is noise. The C++ macro
// parser folds the same case (parse_expression.hpp).
function toDrawNode(n, featureNames) {
  switch (n.type) {
    case "num":
      return { kind: "constant", label: constLabel(n.value), children: [] };
    case "var":
      return { kind: "variable", label: varLabel(n.index, featureNames), children: [] };
    case "neg":
      if (n.a.type === "num")
        return { kind: "constant", label: constLabel(-n.a.value), children: [] };
      // Labelled with the sign the expression string shows, not the engine's "neg": one
      // operator must not appear under two names on one screen. A one-child "-" is
      // unambiguous beside the two-child subtraction.
      return { kind: "operator", label: "-", children: [toDrawNode(n.a, featureNames)] };
    case "call":
      return { kind: "operator", label: n.name, children: [toDrawNode(n.a, featureNames)] };
    case "bin":
      return {
        kind: "operator",
        label: BINARY_LABEL[n.op] || n.op,
        children: [toDrawNode(n.a, featureNames), toDrawNode(n.b, featureNames)],
      };
    default:
      throw new Error(`Unknown node type ${n.type}`);
  }
}

// --- Layout -----------------------------------------------------------------------

// Leaves take consecutive integer columns; an inner node sits at the mean of its children.
// Sibling subtrees own disjoint leaf ranges, so two nodes at the same depth can never
// collide, and a unary node lands directly above its only child. Reingold-Tilford would
// tighten wide trees, but at the default max_nodes = 30 this is visually the same picture
// for a fraction of the code.
function layout(root) {
  const nodes = [];
  let nextLeaf = 0;
  (function walk(n, depth, parent) {
    const rec = { id: nodes.length, parent, depth, x: 0, label: n.label, kind: n.kind };
    nodes.push(rec);
    if (n.children.length === 0) {
      rec.x = nextLeaf++;
    } else {
      const xs = n.children.map((c) => walk(c, depth + 1, rec.id));
      rec.x = xs.reduce((s, v) => s + v, 0) / xs.length;
    }
    return rec.x;
  })(root, 0, null);
  return nodes;
}

// Public: the flat node list for one expression string. Exported for direct testing and
// reuse; renderTree() is the normal entry point.
export function buildTree(expr, featureNames) {
  return layout(toDrawNode(parseExpression(String(expr)), featureNames));
}

// --- Rendering --------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";
const el = (name, attrs) => {
  const node = document.createElementNS(SVG_NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  return node;
};

// Draw `expr` into `container` as one inline <svg>, replacing whatever was there.
//
// Colours come from the theme's CSS variables through the stylesheet (class names on the
// nodes), never from JavaScript: the theme toggle then recolours the tree with no redraw,
// unlike the Chart.js plots which must be rebuilt (plots.js themeColors).
//
// Widths are measured, not estimated: the text is inserted first, getComputedTextLength()
// read, and only then are the capsules and the viewBox sized. The column pitch uses the
// widest capsule in the tree, so a long constant such as "-1.23457e-05" pushes its
// neighbours apart instead of overlapping them.
export function renderTree(container, expr, featureNames) {
  container.textContent = "";
  let nodes;
  try {
    nodes = buildTree(expr, featureNames);
  } catch (e) {
    // A tree is a secondary reading of the equation; a parse failure must never blank or
    // break the results panel.
    container.textContent = String(expr);
    return null;
  }

  const svg = el("svg", { class: "tree-svg", xmlns: SVG_NS, "font-size": FONT_SIZE });
  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = `Equation tree: ${expr}`;
  svg.appendChild(title);
  svg.setAttribute("role", "img");

  const edges = el("g", { class: "tree-edges" });
  const groups = el("g", { class: "tree-nodes" });
  svg.appendChild(edges);
  svg.appendChild(groups);
  container.appendChild(svg);

  // Pass 1: text only, so the browser can measure it.
  const texts = nodes.map((n) => {
    const g = el("g", { class: `tree-node ${n.kind}` });
    const rect = el("rect", { rx: NODE_H / 2, ry: NODE_H / 2, height: NODE_H });
    const text = el("text", { "text-anchor": "middle", "dominant-baseline": "central" });
    text.textContent = n.label;
    g.appendChild(rect);
    g.appendChild(text);
    groups.appendChild(g);
    return { rect, text };
  });

  // Pass 2: size every capsule, then place them on a pitch set by the widest one.
  let maxW = NODE_H;
  nodes.forEach((n, i) => {
    const measured = texts[i].text.getComputedTextLength();
    const w = measured > 0 ? measured : n.label.length * FALLBACK_CHAR_W;
    n.w = Math.max(NODE_H, w + 2 * PAD_X);
    if (n.w > maxW) maxW = n.w;
  });

  const pitch = maxW + GAP;
  const depth = nodes.reduce((m, n) => Math.max(m, n.depth), 0);
  const cols = nodes.reduce((m, n) => Math.max(m, n.x), 0);
  const width = 2 * MARGIN + maxW + cols * pitch;
  const height = 2 * MARGIN + NODE_H + depth * LEVEL_H;

  nodes.forEach((n, i) => {
    n.cx = MARGIN + maxW / 2 + n.x * pitch;
    n.cy = MARGIN + NODE_H / 2 + n.depth * LEVEL_H;
    const { rect, text } = texts[i];
    rect.setAttribute("x", n.cx - n.w / 2);
    rect.setAttribute("y", n.cy - NODE_H / 2);
    rect.setAttribute("width", n.w);
    text.setAttribute("x", n.cx);
    text.setAttribute("y", n.cy);
  });

  // Edges are drawn from capsule edge to capsule edge, not centre to centre, so a line
  // never shows through a node.
  nodes.forEach((n) => {
    if (n.parent == null) return;
    const p = nodes[n.parent];
    edges.appendChild(el("line", {
      x1: p.cx, y1: p.cy + NODE_H / 2, x2: n.cx, y2: n.cy - NODE_H / 2,
    }));
  });

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  return svg;
}

// Serialise a rendered tree as a standalone .svg file. The on-page SVG takes its colours
// from CSS variables, which do not exist outside the document, so every painted property is
// resolved and written onto the clone as a presentation attribute.
const INLINED_PROPS = ["fill", "stroke", "stroke-width", "font-family", "font-size"];
export function treeSvgStandalone(svg) {
  const clone = svg.cloneNode(true);
  const source = svg.querySelectorAll("rect, line, text");
  const target = clone.querySelectorAll("rect, line, text");
  source.forEach((node, i) => {
    const cs = getComputedStyle(node);
    INLINED_PROPS.forEach((p) => target[i].setAttribute(p, cs.getPropertyValue(p)));
  });
  clone.setAttribute("xmlns", SVG_NS);
  return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
}
