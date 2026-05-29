"""Fonctions de récompense pour le GRPO sur GSM8K.

Chaque reward reçoit `completions` (+ les colonnes du dataset en kwargs, ici
`answer` = champ réponse GSM8K brut avec '#### N') et renvoie une liste de floats.
TRL somme les rewards (pondérées par reward_weights).

On cible explicitement les deux faiblesses observées sur le baseline :
- justesse du résultat final  -> correctness_reward
- respect du format '#### N'   -> format_reward (88 % des sorties baseline
  n'avaient PAS de '####' !)
"""

import re

from answer_utils import (
    _GOLD_RE,
    extract_gold_answer,
    extract_pred_answer,
    is_exact_match,
)

_NUM_AFTER_HASH = re.compile(r"####\s*-?\d")


def _text(completion):
    """completion peut être une chaîne ou une liste de messages (conversationnel)."""
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def correctness_reward(completions, answer, **kwargs):
    """1.0 si la réponse finale (après '####') == gold, sinon 0.0."""
    rewards = []
    for comp, ans in zip(completions, answer):
        pred = extract_pred_answer(_text(comp))
        gold = extract_gold_answer(ans)
        rewards.append(1.0 if is_exact_match(pred, gold) else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    """1.0 si la sortie contient bien une ligne '#### <nombre>', sinon 0.0."""
    out = []
    for comp in completions:
        t = _text(comp)
        ok = bool(_GOLD_RE.search(t)) and extract_pred_answer(t) != ""
        out.append(1.0 if ok else 0.0)
    return out
