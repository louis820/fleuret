"""Éval GSM8K avec tool calling (calculatrice), comparable au baseline.

Le modèle peut déléguer l'arithmétique : il écrit une expression entre
<calc> et </calc> et s'arrête ; on évalue (calculator.safe_calc) et on injecte
<result>...</result>, puis il continue. Métrique identique : exact match sur la
réponse après '####'. Logge dans MLflow (même expérience que le baseline).

Boucle ReAct robuste — voir les edge-cases gérés dans le code et calculator.py.

⚠️ Plus lent que le baseline : chaque tool call = une génération supplémentaire
   (le préfixe est ré-encodé à chaque tour).

Usage:
    uv run baseline_tool_call.py --n 100
    uv run baseline_tool_call.py --n 5 --verbose
"""

import argparse
import json
import re
import time

import mlflow
import torch

from answer_utils import extract_gold_answer, extract_pred_answer, is_exact_match
from calculator import safe_calc
from load_gsm8k import load_gsm8k
from load_model import MODEL_NAME, load_model

EXPERIMENT_NAME = "gsm8k-baseline"   # même expérience → comparaison directe

CALC_OPEN, CALC_CLOSE = "<calc>", "</calc>"

SYSTEM_PROMPT = (
    "You are a math assistant with access to a calculator tool.\n"
    "When you need to compute an arithmetic expression, write it between "
    "<calc> and </calc> and STOP. Use only numbers and the operators + - * / ( ). "
    "Do NOT put '=' or variables inside. The result will be inserted as "
    "<result>...</result>; then continue.\n"
    "Always give the final numeric answer on a new line after '#### '.\n\n"
    "Example:\n"
    "Question: A box has 3 rows of 24 apples. How many apples in total?\n"
    "We multiply rows by apples per row: <calc>3*24</calc> <result>72</result>\n"
    "So there are 72 apples.\n"
    "#### 72"
)

_CALC_RE = re.compile(re.escape(CALC_OPEN) + r"(.*?)" + re.escape(CALC_CLOSE), re.S)


def generate_with_tool(model, tokenizer, device, question, max_new_tokens, max_tool_calls):
    """Boucle ReAct : génère, exécute les <calc>, retourne (transcript, n_tool_calls)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    transcript = ""      # texte côté assistant (génération + résultats injectés)
    tool_calls = 0

    # +1 itération pour la génération finale (réponse après le dernier outil).
    for _ in range(max_tool_calls + 1):
        allow_tool = tool_calls < max_tool_calls
        inputs = tokenizer([prompt + transcript], return_tensors="pt").to(device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if allow_tool:
            # S'arrête dès qu'un appel outil est complet.
            gen_kwargs.update(stop_strings=[CALC_CLOSE], tokenizer=tokenizer)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        transcript += new

        # A-t-on un appel outil complet à exécuter ?
        if allow_tool and new.rstrip().endswith(CALC_CLOSE):
            m = _CALC_RE.search(new)
            expr = m.group(1) if m else ""
            result = safe_calc(expr)
            transcript += f" <result>{result}</result> "
            tool_calls += 1
            continue

        # Sinon : réponse finale (ou plus de budget outil) → on s'arrête.
        break

    return transcript, tool_calls


def evaluate(n, max_new_tokens, max_tool_calls, verbose, adapter, quant):
    model, tokenizer, device = load_model(adapter=adapter, quant=quant)
    gsm8k = load_gsm8k()
    test_set = gsm8k["test"].select(range(min(n, len(gsm8k["test"]))))
    n = len(test_set)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="tool-call"):
        mlflow.log_params({
            "model": MODEL_NAME,
            "adapter": adapter or "none",
            "quant": quant or "none",
            "mode": "tool-call",
            "max_tool_calls": max_tool_calls,
            "n_questions": n,
            "max_new_tokens": max_new_tokens,
            "device": device,
            "decoding": "greedy",
        })

        records, correct, total_tool_calls = [], 0, 0
        t0 = time.time()

        for i, example in enumerate(test_set):
            question = example["question"]
            gold = extract_gold_answer(example["answer"])

            transcript, n_calls = generate_with_tool(
                model, tokenizer, device, question, max_new_tokens, max_tool_calls
            )
            pred = extract_pred_answer(transcript)
            ok = is_exact_match(pred, gold)
            correct += int(ok)
            total_tool_calls += n_calls

            records.append({
                "idx": i, "question": question, "gold": gold, "pred": pred,
                "correct": ok, "n_tool_calls": n_calls, "generation": transcript,
            })
            running_acc = correct / (i + 1)
            if verbose:
                print("\n" + "=" * 80)
                print(f"[{i + 1}/{n}] ok={ok} calls={n_calls} (gold={gold!r} pred={pred!r})")
                print("-" * 80)
                print("QUESTION:\n" + question)
                print("-" * 80)
                print("TRANSCRIPT:\n" + transcript)
            else:
                print(f"[{i + 1}/{n}] gold={gold!r} pred={pred!r} ok={ok} "
                      f"calls={n_calls} | acc={running_acc:.3f}")
            mlflow.log_metric("running_accuracy", running_acc, step=i)

        elapsed = time.time() - t0
        accuracy = correct / n

        mlflow.log_metrics({
            "exact_match": accuracy,
            "n_correct": correct,
            "elapsed_seconds": elapsed,
            "seconds_per_question": elapsed / n,
            "avg_tool_calls": total_tool_calls / n,
            "questions_using_tool": sum(1 for r in records if r["n_tool_calls"] > 0),
        })

        preds_path = "predictions_tool.jsonl"
        with open(preds_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mlflow.log_artifact(preds_path)

        print(f"\n=== Exact match: {accuracy:.3f} ({correct}/{n}) en {elapsed:.1f}s ===")
        print(f"appels outil moyens/question : {total_tool_calls / n:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Éval GSM8K + tool calling (calculatrice)")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-tool-calls", type=int, default=8,
                   help="nb max d'appels calculatrice par question (anti-boucle)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--adapter", default=None)
    p.add_argument("--quant", choices=["nf4", "fp4", "8bit"], default=None)
    args = p.parse_args()
    evaluate(args.n, args.max_new_tokens, args.max_tool_calls,
             args.verbose, args.adapter, args.quant)


if __name__ == "__main__":
    main()
