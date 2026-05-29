"""GRPO sur Google Colab T4 (Turing, 16 Go, fp16 only) — GSM8K, Qwen2.5-1.5B.

Spécificités T4 vs train_grpo.py :
- fp16 forcé (le T4 n'a pas de bf16).
- PAS de vLLM (fragile sur Turing) → génération HF.
- gradient checkpointing ON (tenir dans 16 Go).
- checkpoints FRÉQUENTS + reprise après coupure de session Colab.

Réutilise les briques de train_grpo.py (rewards, dataset, rollout tool).

--- Sur Colab ---
    from google.colab import drive; drive.mount('/content/drive')
    !cd /content/fleuret && uv run train_grpo_t4.py \
        --tool --output-dir /content/drive/MyDrive/grpo-gsm8k-t4

    # après une déconnexion, RELANCER la MÊME commande avec --resume :
    !cd /content/fleuret && uv run train_grpo_t4.py \
        --tool --output-dir /content/drive/MyDrive/grpo-gsm8k-t4 --resume
"""

import argparse
import os

import mlflow
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from grpo_reward import correctness_reward, format_reward
from load_model import MODEL_NAME
from train_grpo import (
    EXPERIMENT_NAME,
    SYSTEM_PLAIN,
    SYSTEM_TOOL,
    TARGET_MODULES,
    build_dataset,
    make_tool_rollout,
)


def latest_checkpoint(output_dir):
    """Dernier checkpoint-* présent (pour reprise)."""
    if not os.path.isdir(output_dir):
        return None
    cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not cks:
        return None
    return os.path.join(output_dir, max(cks, key=lambda d: int(d.split("-")[1])))


def main() -> None:
    p = argparse.ArgumentParser(description="GRPO GSM8K sur T4 (Colab)")
    p.add_argument("--tool", action="store_true", help="tool-calling calculatrice (HF gen)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=150)
    # --- batch T4 : voir le commentaire en bas du fichier ---
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4, help="par device (multiple de num_generations)")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-prompt-length", type=int, default=384)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--max-tool-calls", type=int, default=4)
    p.add_argument("--save-steps", type=int, default=25, help="checkpoint fréquent (Colab)")
    p.add_argument("--save-total-limit", type=int, default=3, help="ne pas saturer Drive")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--output-dir", default="grpo-gsm8k-t4",
                   help="POINTER SUR GOOGLE DRIVE pour survivre aux coupures")
    p.add_argument("--resume", action="store_true", help="reprendre du dernier checkpoint")
    args = p.parse_args()

    system_prompt = SYSTEM_TOOL if args.tool else SYSTEM_PLAIN
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = build_dataset(system_prompt, args.max_train)
    print(f"Dataset GRPO : {len(train_ds)} prompts | mode={'tool' if args.tool else 'plain'}")

    lora = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        fp16=True,                       # T4 : fp16, surtout PAS bf16
        bf16=False,
        gradient_checkpointing=True,     # tenir dans 16 Go
        use_vllm=False,                  # Turing : pas de vLLM
        reward_weights=[1.0, 0.3],
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="mlflow",
        run_name=f"t4-grpo-{'tool' if args.tool else 'plain'}",
    )

    rollout_func = (
        make_tool_rollout(tokenizer, args.max_tool_calls, args.max_completion_length)
        if args.tool else None
    )

    resume = latest_checkpoint(args.output_dir) if args.resume else None
    if args.resume:
        print(f"Reprise depuis : {resume or '(aucun checkpoint trouvé → départ à zéro)'}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params({
            "base_model": MODEL_NAME, "device": "T4", "precision": "fp16",
            "mode": "tool" if args.tool else "plain", "lora_rank": args.rank,
            "max_steps": args.max_steps, "num_generations": args.num_generations,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "max_completion_length": args.max_completion_length, "lr": args.lr,
        })

        kw = dict(model=MODEL_NAME, reward_funcs=[correctness_reward, format_reward],
                  args=cfg, train_dataset=train_ds, peft_config=lora)
        if rollout_func is not None:
            kw["rollout_func"] = rollout_func

        trainer = GRPOTrainer(**kw)
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nAdapter GRPO (T4) sauvegardé dans : {args.output_dir}")


# --- BATCH T4 ----------------------------------------------------------------
# per_device_train_batch_size doit être un multiple de num_generations.
# Défaut : num_generations=4, batch=4 (→ 1 prompt unique × 4 complétions par
# micro-step), grad_accum=4 → 16 complétions / step optimiseur = 4 prompts.
# C'est le réglage tenable sur 16 Go. Pour aller plus vite en mode standard
# (sans --tool), on peut monter batch=8 (2 prompts × 4).
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
