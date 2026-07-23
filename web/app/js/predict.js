// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Browser-side expression evaluator. `predict` is not in the C++ core or the bindings:
// the R and Python packages each re-parse the returned infix string (R with its own
// operators, Python via `eval` in a restricted NumPy namespace — see
// rsymbolic2/__init__.py::_eval_expression). We do the same here with a small, safe
// recursive-descent parser (no `eval`), matching the grammar emitted by
// tree.hpp::to_string: variables `x<i>`, `%.6g` constants (incl. inf / nan /
// scientific), unary `name(arg)` for neg/exp/log/sin/cos/sqrt/tanh/abs/square/inv, and
// fully-parenthesised binary `(a op b)` with op in + - * / ^.
//
// Safe-pow caveat (identical to the Python wrapper): `^` maps to JS `**`, which yields
// NaN for a negative base with a non-integer exponent — this differs from the
// training-time safe pow (which returns 0 there). It only matters if the fitted
// expression raises a negative base to a fractional power on the prediction inputs.

const UNARY_FNS = {
  neg: (v) => -v,
  square: (v) => v * v,
  inv: (v) => 1 / v,
  exp: Math.exp,
  log: Math.log, // natural log, matching the core
  sin: Math.sin,
  cos: Math.cos,
  sqrt: Math.sqrt,
  tanh: Math.tanh,
  abs: Math.abs,
};

// --- Tokenizer --------------------------------------------------------------------
function tokenize(s) {
  const tokens = [];
  let i = 0;
  const isDigit = (c) => c >= "0" && c <= "9";
  const isIdentStart = (c) => (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || c === "_";
  const isIdent = (c) => isIdentStart(c) || isDigit(c);
  while (i < s.length) {
    const c = s[i];
    if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      i++;
      continue;
    }
    if ("+-*/^()".includes(c)) {
      tokens.push({ type: "op", value: c });
      i++;
      continue;
    }
    // Number: digits with optional decimal point and exponent. A leading sign is handled
    // by the parser (unary minus), not here, so "1.23e-05" tokenizes as one number but
    // "-1.3" is unary-minus applied to 1.3.
    if (isDigit(c) || (c === "." && isDigit(s[i + 1] || ""))) {
      let j = i;
      while (j < s.length && isDigit(s[j])) j++;
      if (s[j] === ".") {
        j++;
        while (j < s.length && isDigit(s[j])) j++;
      }
      if (s[j] === "e" || s[j] === "E") {
        j++;
        if (s[j] === "+" || s[j] === "-") j++;
        while (j < s.length && isDigit(s[j])) j++;
      }
      tokens.push({ type: "num", value: parseFloat(s.slice(i, j)) });
      i = j;
      continue;
    }
    if (isIdentStart(c)) {
      let j = i;
      while (j < s.length && isIdent(s[j])) j++;
      const word = s.slice(i, j);
      if (word === "inf" || word === "Inf") tokens.push({ type: "num", value: Infinity });
      else if (word === "nan" || word === "NaN") tokens.push({ type: "num", value: NaN });
      else tokens.push({ type: "ident", value: word });
      i = j;
      continue;
    }
    throw new Error(`Unexpected character '${c}' at position ${i} in "${s}"`);
  }
  return tokens;
}

// --- Parser (recursive descent) ---------------------------------------------------
// expr   := term (('+'|'-') term)*
// term   := factor (('*'|'/') factor)*
// factor := unary ('^' factor)?      (right-associative power)
// unary  := '-' unary | primary
// primary:= num | 'x'<i> | funcname '(' expr ')' | '(' expr ')'
function parse(str) {
  const tokens = tokenize(str);
  let pos = 0;
  const peek = () => tokens[pos];
  const next = () => tokens[pos++];
  const expectOp = (v) => {
    const t = next();
    if (!t || t.type !== "op" || t.value !== v)
      throw new Error(`Expected '${v}' in "${str}"`);
  };

  function parseExpr() {
    let node = parseTerm();
    while (peek() && peek().type === "op" && (peek().value === "+" || peek().value === "-")) {
      const op = next().value;
      const rhs = parseTerm();
      node = { type: "bin", op, a: node, b: rhs };
    }
    return node;
  }
  function parseTerm() {
    let node = parseFactor();
    while (peek() && peek().type === "op" && (peek().value === "*" || peek().value === "/")) {
      const op = next().value;
      const rhs = parseFactor();
      node = { type: "bin", op, a: node, b: rhs };
    }
    return node;
  }
  function parseFactor() {
    const node = parseUnary();
    if (peek() && peek().type === "op" && peek().value === "^") {
      next();
      const rhs = parseFactor(); // right-associative
      return { type: "bin", op: "^", a: node, b: rhs };
    }
    return node;
  }
  function parseUnary() {
    if (peek() && peek().type === "op" && peek().value === "-") {
      next();
      return { type: "neg", a: parseUnary() };
    }
    return parsePrimary();
  }
  function parsePrimary() {
    const t = next();
    if (!t) throw new Error(`Unexpected end of expression in "${str}"`);
    if (t.type === "num") return { type: "num", value: t.value };
    if (t.type === "op" && t.value === "(") {
      const node = parseExpr();
      expectOp(")");
      return node;
    }
    if (t.type === "ident") {
      // Variable x<i>
      const m = /^x(\d+)$/.exec(t.value);
      if (m) return { type: "var", index: parseInt(m[1], 10) };
      // Unary function call name(arg)
      if (Object.prototype.hasOwnProperty.call(UNARY_FNS, t.value)) {
        expectOp("(");
        const arg = parseExpr();
        expectOp(")");
        return { type: "call", name: t.value, a: arg };
      }
      throw new Error(`Unknown symbol '${t.value}' in "${str}"`);
    }
    throw new Error(`Unexpected token '${t.value}' in "${str}"`);
  }

  const ast = parseExpr();
  if (pos !== tokens.length) throw new Error(`Trailing tokens in "${str}"`);
  return ast;
}

// --- Evaluator --------------------------------------------------------------------
function evalNode(node, row) {
  switch (node.type) {
    case "num":
      return node.value;
    case "var":
      return row[node.index];
    case "neg":
      return -evalNode(node.a, row);
    case "call":
      return UNARY_FNS[node.name](evalNode(node.a, row));
    case "bin": {
      const a = evalNode(node.a, row);
      const b = evalNode(node.b, row);
      switch (node.op) {
        case "+": return a + b;
        case "-": return a - b;
        case "*": return a * b;
        case "/": return a / b;
        case "^": return a ** b;
        default: throw new Error(`Unknown operator ${node.op}`);
      }
    }
    default:
      throw new Error(`Unknown node type ${node.type}`);
  }
}

// Public API ----------------------------------------------------------------------

// Parse an infix expression string into a reusable AST.
export function parseExpression(str) {
  return parse(str);
}

// Evaluate a parsed AST (or a string) on a 2-D array X (array of row arrays).
// Returns a Float64Array of length X.length.
export function predict(exprOrAst, X) {
  const ast = typeof exprOrAst === "string" ? parse(exprOrAst) : exprOrAst;
  const out = new Float64Array(X.length);
  for (let i = 0; i < X.length; i++) out[i] = evalNode(ast, X[i]);
  return out;
}
