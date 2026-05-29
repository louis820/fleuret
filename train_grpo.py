"""GRPO sur GSM8K (Qwen2.5-1.5B-Instruct), pensé pour une L4 (24 Go, bf16).

Récompense = justesse de la réponse finale (#### N) + respect du format.
On entraîne le modèle à PRODUIRE la bonne réponse dans son propre style
(pas d'imitation du dataset) → évite le catastrophic forgetting du SFT.

Deux modes :
- défaut         : GRPO standard. --vllm pour accélérer la génération.
- --tool         : tool-calling EN BOUCLE via rollout_func (calculatrice).
                   Génération HF (pas vLLM dans ce mode). ⚠️ EXPÉRIMENTAL :
                   rollout_func est marqué expérimental dans TRL et les tokens
                   <result> injectés ne sont pas masqués (logprob 0.0).
                   À VALIDER sur un run court avant un vrai run.

Stack requis (résolu dans uv.lock) : transformers 5.9 + trl 1.5.1 (rollout_func)
+ vllm 0.22 (pour --vllm). bitsandbytes/vllm = Linux/CUDA → lancer sur GPU.

Lancer (L4) :
    uv run train_grpo.py --max-steps 300 --vllm        # GRPO standard, rapide
    uv run train_grpo.py --tool --max-steps 300        # avec calculatrice (HF gen)

Suivi : MLflow (report_to), expérience "gsm8k-grpo".
"""

import argparse

import mlflow
import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from calculator import safe_calc
from data_prep import load_splits
from grpo_reward import correctness_reward, format_reward
from load_model import MODEL_NAME

EXPERIMENT_NAME = "gsm8k-grpo"

CALC_OPEN, CALC_CLOSE = "<calc>", "</calc>"

SYSTEM_PLAIN = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then give the final numeric answer on a new line after '#### '."
)

SYSTEM_TOOL = (
    "You are a math assistant with access to a calculator tool.\n"
    "When you need to compute an arithmetic expression, write it between "
    "<calc> and </calc> and STOP. Use only numbers and the operators + - * / ( ). "
    "Do NOT put '=' or variables inside. The result will be inserted as "
    "<result>...</result>; then continue.\n"
    "Always give the final numeric answer on a new line after '#### '."
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def build_dataset(system_prompt, max_train):
    """Dataset GRPO : colonne 'prompt' (messages) + 'answer' (gold brut)."""
    train_set, _ = load_splits()
    if max_train is not None:
        train_set = train_set.select(range(min(max_train, len(train_set))))

    def to_grpo(ex):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": ex["answer"],   # champ GSM8K brut (avec '#### N')
        }

    return train_set.map(to_grpo, remove_columns=train_set.column_names)


# --- Tool-calling rollout (EXPÉRIMENTAL) ------------------------------------
def make_tool_rollout(tokenizer, max_tool_calls, max_completion_length):
    """rollout_func TRL exécutant la calculatrice en boucle (génération HF).

    Signature TRL : rollout_func(prompts, trainer) -> dict avec les clés
    'prompt_ids', 'completion_ids', 'logprobs' (une liste par complétion).

    On génère via `trainer.model` (politique courante) plutôt que via le moteur
    vLLM colocate de TRL, qui est mis en sleep/wake + resync de poids et donc
    impossible à piloter proprement depuis ici. Conséquence : plus lent, mais
    correct et indépendant de la version.

    ⚠️ LIMITES (expérimental, à valider sur un run court) :
    - les tokens <result> injectés reçoivent un logprob 0.0 (non échantillonnés)
      → biais sur le ratio d'importance pour ces (rares, courts) tokens ;
    - les logprobs viennent de output_scores (logits post-processing), proches
      mais pas strictement la distribution d'échantillonnage avec température.
    """
    import torch

    def rollout_func(prompts, trainer, **kwargs):
        model = trainer.model
        device = next(model.parameters()).device
        num_gen = trainer.num_generations
        temp = trainer.args.temperature
        eos_id = tokenizer.eos_token_id

        prompt_ids_all, completion_ids_all, logprobs_all = [], [], []
        was_training = model.training
        model.eval()
        try:
            for prompt in prompts:
                base_text = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                base_ids = tokenizer(base_text, add_special_tokens=False).input_ids
                for _ in range(num_gen):
                    comp_ids, comp_lps, text = [], [], base_text
                    for turn in range(max_tool_calls + 1):
                        allow_tool = turn < max_tool_calls
                        budget = max_completion_length - len(comp_ids)
                        if budget <= 0:
                            break
                        enc = tokenizer(text, add_special_tokens=False,
                                        return_tensors="pt").to(device)
                        gk = dict(max_new_tokens=budget, return_dict_in_generate=True,
                                  output_scores=True, pad_token_id=eos_id,
                                  do_sample=temp > 0)
                        if temp > 0:
                            gk["temperature"] = temp
                        if allow_tool:
                            gk.update(stop_strings=[CALC_CLOSE], tokenizer=tokenizer)
                        with torch.no_grad():
                            out = model.generate(**enc, **gk)
                        new_ids = out.sequences[0][enc.input_ids.shape[1]:].tolist()
                        for tok, score in zip(new_ids, out.scores):
                            comp_lps.append(
                                torch.log_softmax(score[0].float(), -1)[tok].item()
                            )
                        comp_ids += new_ids
                        new_text = tokenizer.decode(new_ids, skip_special_tokens=False)
                        text += new_text
                        if allow_tool and new_text.rstrip().endswith(CALC_CLOSE):
                            expr = new_text.rsplit(CALC_OPEN, 1)[-1].split(CALC_CLOSE)[0]
                            injected = f" <result>{safe_calc(expr)}</result> "
                            inj_ids = tokenizer(injected, add_special_tokens=False).input_ids
                            comp_ids += inj_ids
                            comp_lps += [0.0] * len(inj_ids)
                            text += injected
                            continue
                        break
                    prompt_ids_all.append(base_ids)
                    completion_ids_all.append(comp_ids[:max_completion_length])
                    logprobs_all.append(comp_lps[:max_completion_length])
        finally:
            if was_training:
                model.train()

        return {
            "prompt_ids": prompt_ids_all,
            "completion_ids": completion_ids_all,
            "logprobs": logprobs_all,
        }

    return rollout_func


def main() -> None:
    p = argparse.ArgumentParser(description="GRPO GSM8K + MLflow (L4)")
    p.add_argument("--tool", action="store_true", help="tool-calling en boucle (expérimental)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8, help="par device (multiple de num_generations)")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04, help="coeff KL")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--max-tool-calls", type=int, default=6)
    p.add_argument("--gpu-mem-util", type=float, default=0.35, help="part VRAM pour vLLM (colocate)")
    p.add_argument("--vllm", action="store_true",
                   help="génération via vLLM (rapide) — NÉCESSITE un venv transformers<4.56")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--output-dir", default="grpo-gsm8k")
    p.add_argument("--no-bf16", action="store_true", help="fp16 (ex: GPU Turing/T4)")
    args = p.parse_args()

    # Le rollout tool gère lui-même la génération (HF) → on désactive vLLM
    # dans ce mode (vLLM ne sert que pour la génération GRPO standard).
    use_vllm = args.vllm and not args.tool
    if args.tool and args.vllm:
        print("Note: --tool gère la génération via HF → vLLM désactivé pour ce run.")

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
        bf16=not args.no_bf16,
        fp16=args.no_bf16,
        use_vllm=use_vllm,
        vllm_mode="colocate",                 # vLLM sur le même GPU (L4 unique)
        vllm_gpu_memory_utilization=args.gpu_mem_util,
        reward_weights=[1.0, 0.3],            # justesse domine, format en appoint
        logging_steps=5,
        save_steps=50,
        report_to="mlflow",
        run_name=f"grpo-{'tool' if args.tool else 'plain'}-r{args.rank}",
    )

    rollout_func = None
    if args.tool:
        rollout_func = make_tool_rollout(
            tokenizer, args.max_tool_calls, args.max_completion_length
        )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params({
            "base_model": MODEL_NAME, "mode": "tool" if args.tool else "plain",
            "lora_rank": args.rank, "max_steps": args.max_steps,
            "num_generations": args.num_generations, "lr": args.lr,
            "beta": args.beta, "temperature": args.temperature,
            "n_train": len(train_ds), "system_prompt": system_prompt,
        })

        trainer_kwargs = dict(
            model=MODEL_NAME,
            reward_funcs=[correctness_reward, format_reward],
            args=cfg,
            train_dataset=train_ds,
            peft_config=lora,
        )
        if rollout_func is not None:
            trainer_kwargs["rollout_func"] = rollout_func

        trainer = GRPOTrainer(**trainer_kwargs)
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nAdapter GRPO sauvegardé dans : {args.output_dir}")


if __name__ == "__main__":
    main()
