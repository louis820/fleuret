"""Explore GSM8K : affiche N échantillons aléatoires du dataset.

Usage:
    uv run explore_gsm8k.py                 # 10 samples du test set
    uv run explore_gsm8k.py --n 5 --split train
    uv run explore_gsm8k.py --seed 42       # reproductible
"""

import argparse
import random

from answer_utils import extract_gold_answer
from load_gsm8k import load_gsm8k


def explore(n: int, split: str, seed: int | None) -> None:
    gsm8k = load_gsm8k()
    data = gsm8k[split]

    rng = random.Random(seed)
    idxs = rng.sample(range(len(data)), k=min(n, len(data)))

    print(f"\n{len(data)} exemples dans '{split}' — affichage de {len(idxs)} au hasard")
    if seed is not None:
        print(f"(seed={seed}, reproductible)")

    for rank, i in enumerate(idxs, 1):
        ex = data[i]
        gold = extract_gold_answer(ex["answer"])
        print("\n" + "=" * 80)
        print(f"#{rank}  [idx {i}]  réponse finale = {gold}")
        print("-" * 80)
        print("Q:", ex["question"])
        print("-" * 80)
        print("A:", ex["answer"])


def main() -> None:
    p = argparse.ArgumentParser(description="Explore GSM8K (samples aléatoires)")
    p.add_argument("--n", type=int, default=10, help="nb d'échantillons")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--seed", type=int, default=None, help="seed pour reproductibilité")
    args = p.parse_args()
    explore(args.n, args.split, args.seed)


if __name__ == "__main__":
    main()
