"""Préparation des données pour le fine-tuning GSM8K.

Stratégie de split :
- ÉVAL : les `N_EVAL` (=100) premières lignes du split `test` — exactement
  celles utilisées par eval_baseline.py.
- TRAIN : le split `train` (7473 lignes).
GSM8K `train` et `test` sont disjoints → les 100 lignes d'éval ne sont JAMAIS
vues à l'entraînement (garanti par construction).

Format SFT : on produit des paires {"prompt", "completion"} pour que le trainer
calcule la loss UNIQUEMENT sur la complétion (la réponse + le raisonnement),
jamais sur le prompt (système + question). On apprend donc au modèle à
*imiter le raisonnement du dataset* et à reproduire la réponse finale après '####'.
"""

import re

from load_gsm8k import load_gsm8k

N_EVAL = 100

# Même prompt système qu'à l'inférence (eval_baseline) → distribution cohérente.
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then give the final numeric answer on a new line after '#### '."
)

# Annotations calculatrice du dataset : "<<4/2=2>>".
_CALC_RE = re.compile(r"<<[^>]*>>")


def load_splits():
    """Retourne (train_set, eval_set) ; eval_set = test[:N_EVAL]."""
    ds = load_gsm8k()
    eval_set = ds["test"].select(range(N_EVAL))
    train_set = ds["train"]
    return train_set, eval_set


def _clean_answer(answer: str, strip_calc: bool) -> str:
    """Nettoie le champ answer ; retire les <<...>> si strip_calc=True."""
    if strip_calc:
        answer = _CALC_RE.sub("", answer)
    return answer.strip()


def build_sft_dataset(tokenizer, strip_calc: bool = False, max_train: int | None = None):
    """Construit le dataset SFT au format prompt/completion.

    - prompt     : template chat (système + question) + balise de génération.
    - completion : la réponse complète du dataset (raisonnement + '#### N') + EOS.
    Le trainer (completion_only_loss=True) masque les tokens du prompt.
    """
    train_set, _ = load_splits()
    if max_train is not None:
        train_set = train_set.select(range(min(max_train, len(train_set))))

    eos = tokenizer.eos_token or "<|im_end|>"

    def to_pair(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion = _clean_answer(example["answer"], strip_calc) + eos
        return {"prompt": prompt, "completion": completion}

    return train_set.map(to_pair, remove_columns=train_set.column_names)


if __name__ == "__main__":
    # Aperçu rapide d'une paire prompt/completion (nécessite le tokenizer).
    from transformers import AutoTokenizer
    from load_model import MODEL_NAME

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = build_sft_dataset(tok, max_train=3)
    print(f"{len(ds)} paires construites\n")
    ex = ds[0]
    print("=== PROMPT (masqué, pas de loss) ===")
    print(ex["prompt"])
    print("\n=== COMPLETION (loss calculée ici) ===")
    print(ex["completion"])
