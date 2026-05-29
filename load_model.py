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


def load_model(model_name: str = MODEL_NAME, adapter: str | None = None,
               quant: str | None = None):
    """Charge le tokenizer et le modèle, retourne (model, tokenizer, device).

    - adapter : chemin d'un adapter LoRA (PEFT) à brancher sur la base.
    - quant   : 'nf4' | 'fp4' | '8bit' | None. Quantif bitsandbytes (CUDA only)
                pour charger la base comme à l'entraînement QLoRA.
    Les imports bitsandbytes/peft sont paresseux → aucun impact sur Mac/MPS.
    """
    device = get_device()
    print(f"Chargement de {model_name} sur {device}"
          + (f" (quant={quant})" if quant else "")
          + (f" + adapter {adapter}" if adapter else "") + " ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant_cfg = None
    if quant in ("nf4", "fp4"):
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    if quant_cfg is not None:
        # device_map="auto" place le modèle quantifié ; pas de .to() ensuite.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_cfg,
            device_map="auto", dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    # device effectif (avec device_map="auto", le modèle est sur cuda:0).
    device = str(next(model.parameters()).device)

    print("Modèle chargé.")
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
