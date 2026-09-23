"""Independent GeneralPoints exact-success verifier."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class GeneralPointsScore:
    correct: bool
    parsed: bool
    formula_valid: bool
    formula: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FORMULA_RE = re.compile(r'["\\']formula["\\']\\s*:\\s*["\\']([^"\\']+)["\\']')


def extract_formula(text: str) -> str | None:
    raw = (text or "").strip()
    candidates = [raw]

    fence = chr(96) * 3
    if fence in raw:
        for chunk in raw.split(fence):
            cleaned = chunk.strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned:
                candidates.append(cleaned)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            formula = payload.get("formula") if isinstance(payload, dict) else None
            if isinstance(formula, str) and formula.strip():
                return formula.strip()
        except Exception:
            pass

    match = _FORMULA_RE.search(raw)
    if match:
        return match.group(1).strip()
    return None


_ALLOWED_BINOPS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
}


def _eval_node(node: ast.AST, numbers: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, numbers)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("only integer constants are allowed")
        numbers.append(int(node.value))
        return Fraction(int(node.value), 1)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left, numbers)
        right = _eval_node(node.right, numbers)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _ALLOWED_BINOPS[type(node.op)](left, right)

    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def score_response(
    response: str,
    display_cards: list[int],
    target: int = 24,
) -> GeneralPointsScore:
    formula = extract_formula(response)
    if formula is None:
        return GeneralPointsScore(
            correct=False,
            parsed=False,
            formula_valid=False,
            formula=None,
            reason="formula_not_found",
        )

    expression = formula.split("=", 1)[0].strip()
    if not expression:
        return GeneralPointsScore(
            correct=False,
            parsed=True,
            formula_valid=False,
            formula=formula,
            reason="empty_formula",
        )

    try:
        tree = ast.parse(expression, mode="eval")
        used_numbers: list[int] = []
        value = _eval_node(tree, used_numbers)
    except Exception:
        return GeneralPointsScore(
            correct=False,
            parsed=True,
            formula_valid=False,
            formula=formula,
            reason="illegal_formula",
        )

    expected = Counter(int(value) for value in display_cards)
    observed = Counter(used_numbers)
    if observed != expected:
        return GeneralPointsScore(
            correct=False,
            parsed=True,
            formula_valid=False,
            formula=formula,
            reason="wrong_number_multiset",
        )

    if value != Fraction(int(target), 1):
        return GeneralPointsScore(
            correct=False,
            parsed=True,
            formula_valid=True,
            formula=formula,
            reason="wrong_target",
        )

    return GeneralPointsScore(
        correct=True,
        parsed=True,
        formula_valid=True,
        formula=formula,
        reason="correct",
    )
