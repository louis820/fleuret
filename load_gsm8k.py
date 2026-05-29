"""Charge le dataset GSM8K (openai/gsm8k, config "main") en local.

Usage:
    ./fleuret/bin/python load_gsm8k.py
"""

from datasets import load_dataset

DATASET_NAME = "openai/gsm8k"
CONFIG = "main"


def load_gsm8k(config: str = CONFIG):
    """Charge GSM8K et retourne le DatasetDict (splits train / test)."""
    print(f"Chargement de {DATASET_NAME} (config={config}) ...")
    ds = load_dataset(DATASET_NAME, config)
    print(ds)
    return ds


if __name__ == "__main__":
    gsm8k = load_gsm8k()

    print(f"\nTrain: {len(gsm8k['train'])} exemples")
    print(f"Test : {len(gsm8k['test'])} exemples")

    example = gsm8k["test"][0]
    print("\n--- Exemple (test[0]) ---")
    print("Question:", example["question"])
    print("Answer  :", example["answer"])
