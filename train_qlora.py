"""Fine-tuning QLoRA de Qwen2.5-1.5B-Instruct sur GSM8K (split train).

- Base chargée en 4-bit (NF4) → QLoRA. Adapters LoRA entraînés en bf16.
- Loss calculée UNIQUEMENT sur la réponse (completion), pas sur la question.
- Logging MLflow automatique (report_to="mlflow").
- L'adapter LoRA est sauvegardé dans --output-dir.

⚠️ Nécessite un GPU NVIDIA (bitsandbytes = CUDA only). À lancer sur la VM L4,
   pas sur Mac/MPS.

Usage (valeurs par défaut raisonnables pour une L4) :
    uv run train_qlora.py
    uv run train_qlora.py --rank 16 --epochs 3 --batch-size 16
    uv run train_qlora.py --rank 8 --max-train 500   # run court de test

Évaluer ensuite l'adapter :
    uv run eval_baseline.py --n 100   # (après avoir branché l'adapter, cf. README)
"""

import argparse

import mlflow
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from data_prep import SYSTEM_PROMPT, build_sft_dataset, load_splits
from gsm8k_eval import score_exact_match
from load_model import MODEL_NAME

EXPERIMENT_NAME = "gsm8k-qlora"

# Modules cibles LoRA : attention + MLP (couverture maximale pour un petit modèle).
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]


class ExactMatchCallback(TrainerCallback):
    """Évalue l'exact-match GSM8K sur un sous-ensemble du holdout d'éval
    (test[:100]) à la fin de chaque epoch, et logge le score dans MLflow.

    Coûteux (génération) : par défaut sur peu de questions. Sur L4, ~10 min
    pour 100 questions sans quantif, davantage en QLoRA → garder eval_n petit.
    """

    def __init__(self, eval_dataset, tokenizer, eval_n, max_new_tokens, every_epochs):
        self.eval_dataset = eval_dataset.select(range(min(eval_n, len(eval_dataset))))
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.every_epochs = every_epochs

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch or 0)
        if self.every_epochs <= 0 or epoch % self.every_epochs != 0:
            return
        res = score_exact_match(
            model, self.tokenizer, self.eval_dataset,
            max_new_tokens=self.max_new_tokens,
        )
        print(
            f"\n[éval epoch {epoch}] exact_match={res['exact_match']:.3f} "
            f"({res['n_correct']}/{res['n']})"
        )
        mlflow.log_metric("eval_exact_match", res["exact_match"], step=state.global_step)
        mlflow.log_metric("eval_n_correct", res["n_correct"], step=state.global_step)


def build_model(quant: str, grad_checkpoint: bool = True):
    """Charge le modèle base. quant ∈ {nf4, fp4, 8bit, none}."""
    if quant in ("nf4", "fp4"):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,   # double quant : ~0.4 bit/param économisés
        )
    elif quant == "8bit":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
    else:  # none → LoRA classique en bf16 (pas de quantification)
        bnb = None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # incompatible avec le gradient checkpointing
    if bnb is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=grad_checkpoint
        )
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning GSM8K + MLflow")
    p.add_argument("--rank", type=int, default=16, help="rang LoRA (r)")
    p.add_argument("--alpha", type=int, default=None, help="lora_alpha (défaut: 2*rank)")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--quant", choices=["nf4", "fp4", "8bit", "none"], default="nf4")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--packing", action="store_true",
                   help="concatène les exemples pour éliminer le padding (~2-3x plus rapide)")
    p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-grad-checkpoint pour gagner ~20-30%% (VRAM ok à 1.5B)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=640, help="longueur max de séquence")
    p.add_argument("--max-train", type=int, default=None, help="sous-échantillon train (debug)")
    p.add_argument("--strip-calc", action="store_true", help="retire les <<...>> du dataset")
    p.add_argument("--output-dir", default="qlora-gsm8k")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=None,
                   help="sauvegarde tous les N steps (sinon: 1x par epoch)")
    # Éval exact-match périodique (génération) sur le holdout test[:100].
    p.add_argument("--eval-n", type=int, default=40,
                   help="nb de questions du holdout évaluées par epoch (0 = désactivé)")
    p.add_argument("--eval-every-epochs", type=int, default=1,
                   help="fréquence de l'éval exact-match, en epochs")
    p.add_argument("--eval-max-new-tokens", type=int, default=512)
    args = p.parse_args()

    alpha = args.alpha if args.alpha is not None else 2 * args.rank

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = build_sft_dataset(
        tokenizer, strip_calc=args.strip_calc, max_train=args.max_train
    )
    print(f"Dataset SFT : {len(train_ds)} paires prompt/completion")

    model = build_model(args.quant, grad_checkpoint=args.grad_checkpoint)

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=alpha,
        lora_dropout=args.dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=args.grad_checkpoint,
        packing=args.packing,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps or 500,
        max_length=args.max_len,
        completion_only_loss=True,   # ← loss sur la réponse uniquement
        report_to="mlflow",          # ← logging MLflow auto
        run_name=f"qlora-r{args.rank}-{args.quant}",
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    # Params lisibles côté MLflow (en plus de ceux loggés par le callback TRL).
    with mlflow.start_run(run_name=sft_cfg.run_name):
        mlflow.log_params({
            "base_model": MODEL_NAME,
            "lora_rank": args.rank,
            "lora_alpha": alpha,
            "lora_dropout": args.dropout,
            "quant": args.quant,
            "target_modules": ",".join(TARGET_MODULES),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "lr": args.lr,
            "max_len": args.max_len,
            "n_train": len(train_ds),
            "strip_calc": args.strip_calc,
            "system_prompt": SYSTEM_PROMPT,
        })

        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_ds,
            peft_config=lora,
            processing_class=tokenizer,
        )

        # Éval exact-match périodique sur le holdout (test[:100]).
        if args.eval_n > 0:
            _, eval_set = load_splits()
            trainer.add_callback(ExactMatchCallback(
                eval_set, tokenizer,
                eval_n=args.eval_n,
                max_new_tokens=args.eval_max_new_tokens,
                every_epochs=args.eval_every_epochs,
            ))

        # Nombre de paramètres entraînables (signal LoRA).
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        print(f"Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.3f} %)")
        mlflow.log_metric("trainable_params", trainable)

        trainer.train()
        trainer.save_model(args.output_dir)   # adapter LoRA + config
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nAdapter sauvegardé dans : {args.output_dir}")


if __name__ == "__main__":
    main()
