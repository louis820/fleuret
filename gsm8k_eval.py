"""Cœur d'évaluation GSM8K (exact match), partagé entre eval_baseline et le
callback d'éval périodique du training.

`score_exact_match` génère une réponse par question et compare la valeur finale
(après '####') à la référence. Renvoie un dict de métriques + les records.
"""

import torch

from answer_utils import extract_gold_answer, extract_pred_answer, is_exact_match
from data_prep import SYSTEM_PROMPT
from load_model import generate


def score_exact_match(model, tokenizer, dataset, max_new_tokens=512,
                      system_prompt=SYSTEM_PROMPT):
    """Évalue `model` sur `dataset` (champs question/answer). Retourne :
    {"exact_match": float, "n_correct": int, "n": int, "records": [...]}.

    Bascule proprement le modèle en mode éval (cache activé) puis restaure son
    état précédent — important quand on l'appelle au milieu d'un training
    (gradient checkpointing + use_cache=False).
    """
    device = next(model.parameters()).device

    was_training = model.training
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    model.config.use_cache = True

    correct = 0
    records = []
    try:
        with torch.no_grad():
            for i, ex in enumerate(dataset):
                gold = extract_gold_answer(ex["answer"])
                generation = generate(
                    model, tokenizer, device, ex["question"],
                    system_prompt=system_prompt, max_new_tokens=max_new_tokens,
                )
                pred = extract_pred_answer(generation)
                ok = is_exact_match(pred, gold)
                correct += int(ok)
                records.append({
                    "idx": i, "question": ex["question"],
                    "gold": gold, "pred": pred,
                    "correct": ok, "generation": generation,
                })
    finally:
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        if was_training:
            model.train()

    n = len(records)
    return {
        "exact_match": correct / n if n else 0.0,
        "n_correct": correct,
        "n": n,
        "records": records,
    }
