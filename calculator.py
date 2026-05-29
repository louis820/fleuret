"""Calculatrice sûre pour le tool calling.

`safe_calc(expr)` évalue une expression arithmétique SANS jamais utiliser eval()
(évaluateur basé sur l'AST, opérateurs whitelistés). Robuste aux sorties bizarres
d'un petit modèle. Retourne une chaîne formatée, ou "ERROR" si non évaluable.
"""

import ast
import operator
import re

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MAX_EXPR_LEN = 200
_MAX_POW_EXP = 100          # borne anti-DoS sur les exposants
_HAS_LETTER = re.compile(r"[a-zA-Z]")


def _clean(expr: str) -> str:
    """Normalise les symboles non-Python avant parsing."""
    expr = expr.strip()
    # Symboles maths usuels → opérateurs Python.
    expr = (expr.replace("×", "*").replace("·", "*").replace("÷", "/")
                .replace("^", "**"))
    # Le modèle écrit parfois "4*2880=11520" : on ne garde que le LHS.
    if "=" in expr:
        expr = expr.split("=", 1)[0]
    # Retire séparateurs de milliers, devises, espaces.
    expr = expr.replace(",", "").replace("$", "").replace("€", "").replace("%", "")
    return expr.strip()


def _eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("constante non numérique")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXP:
            raise ValueError("exposant trop grand")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("nœud non supporté")


def _fmt(x) -> str:
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return str(round(x, 6))
    return str(x)


def safe_calc(expr: str) -> str:
    """Évalue `expr` de façon sûre. Retourne le résultat formaté ou 'ERROR'."""
    cleaned = _clean(expr)
    if not cleaned or len(cleaned) > _MAX_EXPR_LEN or _HAS_LETTER.search(cleaned):
        return "ERROR"
    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _eval(tree.body)
        return _fmt(result)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError,
            OverflowError, RecursionError):
        return "ERROR"


# --- Auto-tests (exécuter : uv run calculator.py) ---------------------------
if __name__ == "__main__":
    cases = {
        "4*2880": "11520",
        "480000/240": "2000",
        "(1/2)*124600": "62300",
        "20/100*400": "80",
        "4*2880=11520": "11520",     # LHS uniquement
        "$1,000 + 200": "1200",      # devise + milliers
        "20×4": "80",                # symbole unicode
        "2^3": "8",                  # ^ → **
        "10/3": "3.333333",          # float arrondi
        "10/0": "ERROR",             # div par zéro
        "4*x": "ERROR",              # variable
        "the total": "ERROR",        # texte
        "": "ERROR",                 # vide
        "9**9**9": "ERROR",          # exposant borné
        "__import__('os')": "ERROR",  # injection
    }
    ok = 0
    for expr, expected in cases.items():
        got = safe_calc(expr)
        flag = "OK " if got == expected else "XX "
        ok += got == expected
        print(f"{flag} safe_calc({expr!r}) = {got!r}  (attendu {expected!r})")
    print(f"\n{ok}/{len(cases)} cas passés")
