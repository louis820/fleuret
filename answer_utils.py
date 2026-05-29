"""Extraction et comparaison des réponses finales GSM8K (format "#### N")."""

import re

# Réponse de référence GSM8K : tout ce qui suit "####".
_GOLD_RE = re.compile(r"####\s*(.+)")
# Nombres dans la sortie du modèle (entiers/décimaux signés, avec séparateurs).
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_number(text: str) -> str:
    """Normalise une réponse numérique : retire $, virgules, espaces, .0 final."""
    text = text.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    # Retire un point décimal sans valeur significative (42.0 -> 42).
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return text


def extract_gold_answer(answer_field: str) -> str:
    """Extrait la réponse de référence d'un champ 'answer' GSM8K (après '####')."""
    match = _GOLD_RE.search(answer_field)
    raw = match.group(1) if match else answer_field
    return normalize_number(raw)


def extract_pred_answer(generation: str) -> str:
    """Extrait la réponse prédite : '####' si présent, sinon dernier nombre."""
    match = _GOLD_RE.search(generation)
    if match:
        return normalize_number(match.group(1))
    numbers = _NUMBER_RE.findall(generation)
    return normalize_number(numbers[-1]) if numbers else ""


def is_exact_match(pred: str, gold: str) -> bool:
    """Exact match sur la réponse finale normalisée."""
    return pred != "" and pred == gold
