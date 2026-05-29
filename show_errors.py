"""Affiche les erreurs depuis predictions.jsonl.

Usage:
    uv run show_errors.py                 # 10 erreurs, avec génération complète
    uv run show_errors.py --n 20          # 20 erreurs
    uv run show_errors.py --short         # juste idx / gold / pred
    uv run show_errors.py --file predictions_baseline.jsonl
"""

import argparse
import json
import textwrap


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="predictions.jsonl")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--short", action="store_true")
    args = p.parse_args()

    with open(args.file) as f:
        records = [json.loads(line) for line in f]
    errs = [r for r in records if not r["correct"]]

    print(f"{len(errs)} erreurs / {len(records)} questions\n")

    for r in errs[: args.n]:
        if args.short:
            print(f"idx {r['idx']:>3}  gold={r['gold']:>8}  pred={r['pred']!r}")
            continue
        print("=" * 80)
        print(f"[idx {r['idx']}]  gold={r['gold']!r}  pred={r['pred']!r}")
        print("-" * 80)
        print("Q:", textwrap.shorten(r["question"], 300))
        print("-" * 80)
        print("GÉNÉRATION:\n" + r["generation"])


if __name__ == "__main__":
    main()
