"""Extraction et comparaison des réponses finales GSM8K (format "#### N")."""

import re

# Réponse de référence GSM8K : tout ce qui suit "####".
_GOLD_RE = re.compile(r"####\s*(.+)")
# Nombres dans la sortie du modèle (entiers/décimaux signés, avec séparateurs).
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_number(text: str) -> str:
    """Normalise une réponse numérique : retire $, virgules, espaces, .0 final."""
    text = text.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    # Décimal : retire les zéros et le point superflus (42.0 -> 42, 2.50 -> 2.5).
    if re.fullmatch(r"-?\d+\.\d+", text):
        text = text.rstrip("0").rstrip(".")
    # Point final isolé : "2." -> "2".
    elif re.fullmatch(r"-?\d+\.", text):
        text = text[:-1]
    return text


def extract_gold_answer(answer_field: str) -> str:
    """Extrait la réponse de référence d'un champ 'answer' GSM8K (après '####')."""
    match = _GOLD_RE.search(answer_field)
    raw = match.group(1) if match else answer_field
    return normalize_number(raw)


def extract_pred_answer(generation: str) -> str:
    """Extrait la réponse prédite depuis la ligne '####' UNIQUEMENT.

    On en prend le premier nombre (gère '#### Final Answer: 15 liters').
    PAS de fallback : si le modèle n'émet pas de '####', la réponse est
    considérée absente ('') → comptée fausse. On évalue ainsi aussi la
    capacité du modèle à respecter le format demandé.
    """
    match = _GOLD_RE.search(generation)
    if match:
        nums = _NUMBER_RE.findall(match.group(1))
        if nums:
            return normalize_number(nums[0])
    return ""


def is_exact_match(pred: str, gold: str) -> bool:
    """Exact match sur la réponse finale normalisée."""
    return pred != "" and pred == gold
