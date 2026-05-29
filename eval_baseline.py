"""Évalue le modèle baseline sur 100 questions du test set GSM8K.

- Métrique : exact match sur la réponse finale (après "####").
- Logge le run dans MLflow (params, métriques, artefact des prédictions).

Usage:
    ./fleuret/bin/python eval_baseline.py
    ./fleuret/bin/python eval_baseline.py --n 100 --max-new-tokens 512

Visualiser les runs :
    ./fleuret/bin/mlflow ui   # puis http://127.0.0.1:5000
"""

import argparse
import json
import time

import mlflow

from answer_utils import extract_gold_answer, extract_pred_answer, is_exact_match
from load_gsm8k import load_gsm8k
from load_model import MODEL_NAME, generate, load_model

EXPERIMENT_NAME = "gsm8k-baseline"

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then give the final numeric answer on a new line after '#### '."
)


def build_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def evaluate(n: int, max_new_tokens: int, verbose: bool = False,
             adapter: str | None = None, quant: str | None = None) -> None:
    model, tokenizer, device = load_model(adapter=adapter, quant=quant)
    gsm8k = load_gsm8k()
    test_set = gsm8k["test"].select(range(min(n, len(gsm8k["test"]))))
    n = len(test_set)

    run_name = "qlora" if adapter else "baseline"
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "adapter": adapter or "none",
                "quant": quant or "none",
                "dataset": "openai/gsm8k",
                "config": "main",
                "split": "test",
                "n_questions": n,
                "max_new_tokens": max_new_tokens,
                "device": device,
                "decoding": "greedy",
            }
        )

        records = []
        correct = 0
        t0 = time.time()

        for i, example in enumerate(test_set):
            question = example["question"]
            gold = extract_gold_answer(example["answer"])

            generation = generate(
                model, tokenizer, device, question,
                system_prompt=SYSTEM_PROMPT, max_new_tokens=max_new_tokens,
            )
            pred = extract_pred_answer(generation)
            ok = is_exact_match(pred, gold)
            correct += int(ok)

            records.append(
                {
                    "idx": i,
                    "question": question,
                    "gold": gold,
                    "pred": pred,
                    "correct": ok,
                    "generation": generation,
                }
            )
            running_acc = correct / (i + 1)
            if verbose:
                print("\n" + "=" * 80)
                print(f"[{i + 1}/{n}]  ok={ok}  (gold={gold!r}  pred={pred!r})")
                print("-" * 80)
                print("QUESTION:\n" + question)
                print("-" * 80)
                print("GÉNÉRATION COMPLÈTE:\n" + generation)
                print("-" * 80)
                print(f"gold={gold!r}  pred={pred!r}  exact_match={ok}  | acc={running_acc:.3f}")
                print("=" * 80)
            else:
                print(f"[{i + 1}/{n}] gold={gold!r} pred={pred!r} ok={ok} | acc={running_acc:.3f}")
            mlflow.log_metric("running_accuracy", running_acc, step=i)

        elapsed = time.time() - t0
        accuracy = correct / n

        mlflow.log_metrics(
            {
                "exact_match": accuracy,
                "n_correct": correct,
                "elapsed_seconds": elapsed,
                "seconds_per_question": elapsed / n,
            }
        )

        # Artefact : toutes les prédictions pour inspection ultérieure.
        preds_path = "predictions.jsonl"
        with open(preds_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mlflow.log_artifact(preds_path)

        print(f"\n=== Exact match: {accuracy:.3f} ({correct}/{n}) en {elapsed:.1f}s ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Éval baseline GSM8K + MLflow")
    parser.add_argument("--n", type=int, default=100, help="nb de questions du test set")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="affiche la question + la génération complète pour chaque sample",
    )
    parser.add_argument("--adapter", default=None,
                        help="chemin d'un adapter LoRA à évaluer (ex: qlora-gsm8k)")
    parser.add_argument("--quant", choices=["nf4", "fp4", "8bit"], default=None,
                        help="quantif de la base (à matcher avec l'entraînement QLoRA)")
    args = parser.parse_args()
    evaluate(args.n, args.max_new_tokens, args.verbose,
             adapter=args.adapter, quant=args.quant)


if __name__ == "__main__":
    main()
