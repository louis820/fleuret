"""Charge un SLM (Qwen2.5-1.5B-Instruct) en local.

Usage:
    ./fleuret/bin/python load_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def get_device() -> str:
    """Sélectionne le meilleur device dispo (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def load_model(model_name: str = MODEL_NAME):
    """Charge le tokenizer et le modèle, retourne (model, tokenizer, device)."""
    device = get_device()
    print(f"Chargement de {model_name} sur {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",  # bfloat16/float16 selon le hardware (transformers >= 5.x)
    ).to(device)
    model.eval()

    print(f"Modèle chargé ({model.num_parameters() / 1e9:.2f} B paramètres).")
    return model, tokenizer, device


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Reason step by step."


def generate(
    model,
    tokenizer,
    device,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 512,
) -> str:
    """Génère une réponse à partir d'un prompt utilisateur (format chat Qwen)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        # Greedy decoding pour une éval déterministe.
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    generated = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer, device = load_model()
    answer = generate(model, tokenizer, device, "What is 17 * 24?")
    print("\n--- Réponse ---")
    print(answer)
