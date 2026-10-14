"""Optional Qwen3.5-0.8B LoRA keep/drop SFT; no model loads on import.

Install the optional training extra, then run ``python -m aerorescue.peft_train
--data records.jsonl --output artifacts/qwen_lora``. Loading is local/cache-only
unless --allow-download is explicit. CUDA is required unless --allow-cpu is set.
This trains a research candidate filter, never an executable rescue policy.
Architecture: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json
PEFT: https://huggingface.co/docs/peft/en/package_reference/lora
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ROLES = ("inspect", "relay", "deliver")
INPUT_FIELDS = {"event", "mission", "evidence", "self", "now_s", "deadline_s",
                "payload_kg", "locations", "duration_s"}


def prompt_messages(local_input: dict, role: str) -> list[dict]:
    """Accept the local input object, never the enclosing labelled record."""
    if role not in ROLES or not isinstance(local_input, dict):
        raise ValueError("Expected a local input dictionary and a supported role")
    if set(local_input) - INPUT_FIELDS:
        raise ValueError("Unexpected input fields; labels and planner metadata must stay outside input")
    decision_input = {key: value for key, value in local_input.items() if key != "event"}
    agent = local_input["self"]
    decision_input["self"] = {key: agent[key] for key in ("position", "capabilities", "soc",
        "payload_kg", "payload_capacity_kg", "speed_m_s") if key in agent}
    commitment = agent.get("commitment")
    decision_input["self"]["commitment"] = None if commitment is None else {
        key: commitment[key] for key in ("protected", "busy_until_s") if key in commitment}
    content = json.dumps({"input": decision_input, "role": role}, sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return [{"role": "system", "content": "Decide whether this agent should remain "
             "a candidate for the requested rescue role using only its local input. "
             "Reply with exactly keep or drop."}, {"role": "user", "content": content}]


def _encoded(tokenizer, local_input, role, target, max_length):
    prefix = tokenizer.apply_chat_template(prompt_messages(local_input, role),
        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    reply = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prefix) + len(reply) > max_length:
        raise ValueError("Input exceeds --max-length; increase it explicitly. No evidence is truncated.")
    return {"input_ids": prefix + reply, "attention_mask": [1] * (len(prefix) + len(reply)),
            "labels": [-100] * len(prefix) + reply}


def _records(path):
    """Honor supplied splits; null labels and evaluation rows never train."""
    partitions, groups = {"train": [], "validation": []}, {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split")
            if split not in (*partitions, "evaluation"):
                raise ValueError("Every record must carry train/validation/evaluation split")
            if not row.get("episode_id"):
                raise ValueError("Every record requires episode_id for split-leakage checks")
            for field in ("episode_id", "sequence_id", "scenario_id"):
                if row.get(field) is not None:
                    key = (field, str(row[field]))
                    if groups.setdefault(key, split) != split:
                        raise ValueError(f"Group {key} occurs in multiple splits")
            if split == "evaluation" or row.get("label") is None:
                continue
            if type(row["label"]) is not int or row["label"] not in (0, 1):
                raise ValueError("Labels must be integer 0, integer 1, or null")
            if row.get("label_source") != "bounded_planner_counterfactual":
                raise ValueError("Only bounded_planner_counterfactual labels are accepted")
            prompt_messages(row["input"], row["role"])
            partitions[split].append(row)
    if not all(partitions.values()):
        raise ValueError("Nonempty labelled train and validation splits are required")
    return partitions


def _load(model_name, revision, allow_cpu, allow_download):
    try:
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Install the training extra: torch>=2.6, transformers>=5.3, "
                           "peft>=0.18.1, accelerate>=1.10") from exc
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA is unavailable; --allow-cpu explicitly permits slow CPU execution")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    options = dict(revision=revision, local_files_only=not allow_download,
                   trust_remote_code=False, token=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Official checkpoint has a vision tower. Feed text only; do not misload it
    # through a generic AutoModelForCausalLM or initialize a random text backbone.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name, dtype=dtype, attn_implementation="sdpa", use_safetensors=True, **options)
    model.config.text_config.use_cache = False
    model.config.use_cache = False
    return torch, tokenizer, model.to(device)


class QwenRoleScorer:
    """Local adapter inference. predict(input, role) returns normalized P(keep).

    This is a two-completion likelihood score, not a calibrated safety probability.
    Caller must retain deterministic feasibility and execution checks.
    """

    def __init__(self, adapter_path, *, model_name=None, allow_cpu=False, allow_download=False):
        from peft import PeftModel

        directory = Path(adapter_path)
        metadata = json.loads((directory / "aerorescue_adapter.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "aerorescue.qwen-role-filter.v1":
            raise ValueError("Unsupported Qwen role-filter adapter format")
        self.threshold = metadata["threshold"]
        self.max_length = metadata["max_length"]
        self.torch, self.tokenizer, base = _load(model_name or metadata["model_name"],
            metadata.get("revision"), allow_cpu, allow_download)
        self.model = PeftModel.from_pretrained(base, str(directory), is_trainable=False,
            local_files_only=True, token=False).eval()

    def predict(self, local_input: dict, role: str) -> float:
        scores = []
        with self.torch.inference_mode():
            for target in ("drop", "keep"):
                sample = _encoded(self.tokenizer, local_input, role, target, self.max_length)
                reply = [value for value in sample["labels"] if value != -100]
                ids = self.torch.tensor([sample["input_ids"]], device=self.model.device)
                logits = self.model(input_ids=ids, attention_mask=self.torch.ones_like(ids),
                    use_cache=False, logits_to_keep=len(reply) + 1).logits[0, :-1].float()
                labels = self.torch.tensor(reply, device=logits.device)
                scores.append(logits.log_softmax(-1).gather(1, labels[:, None]).sum())
        return self.torch.stack(scores).softmax(0)[1].item()

    __call__ = predict


def train(args):
    partitions = _records(args.data)
    if min(args.max_length, args.batch_size, args.rank, args.epochs, args.learning_rate) <= 0:
        raise ValueError("Length, batch size, rank, epochs, and learning rate must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing adapters are never overwritten")
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    torch, tokenizer, model = _load(args.model, args.revision, args.allow_cpu, args.allow_download)
    datasets = {split: [_encoded(tokenizer, row["input"], row["role"],
                "keep" if row["label"] == 1 else "drop", args.max_length) for row in rows]
                for split, rows in partitions.items()}
    # Explicit discovered paths cover attention, DeltaNet, and MLP projections;
    # the vision tower, embeddings, and vocabulary head are excluded.
    targets = [name for name, module in model.named_modules()
               if ".language_model.layers." in name and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("No Qwen3.5 decoder linear layers found; refusing a no-op adapter")
    revision = args.revision or getattr(model.config, "_commit_hash", None)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=args.rank,
        lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none", target_modules=targets))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    trainer = Trainer(model=model, processing_class=tokenizer,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
        args=TrainingArguments(output_dir=str(output), per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1, gradient_accumulation_steps=8,
            num_train_epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=model.dtype == torch.bfloat16, use_cpu=not torch.cuda.is_available(),
            eval_strategy="epoch", save_strategy="no", logging_steps=10,
            prediction_loss_only=True, report_to="none", push_to_hub=False))
    trainer.train()
    metrics = trainer.evaluate()
    model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(output)
    metadata = {"format": "aerorescue.qwen-role-filter.v1", "threshold": 0.5,
                "threshold_calibrated": False,
                "model_name": args.model, "revision": revision, "max_length": args.max_length,
                "seed": args.seed, "train_rows": len(datasets["train"]),
                "validation_rows": len(datasets["validation"]), "validation_metrics": metrics,
                "training_arguments": vars(args),
                "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
                "label_source": "bounded_planner_counterfactual", "evaluation_used": False}
    (output / "aerorescue_adapter.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Grouped, pre-split labelled JSONL")
    parser.add_argument("--output", required=True, help="Empty local adapter output directory")
    parser.add_argument("--model", default=MODEL_ID, help="Official model ID or local snapshot")
    parser.add_argument("--revision", help="Optional pinned Hugging Face commit")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true", help="Permit slow CPU execution")
    parser.add_argument("--allow-download", action="store_true", help="Permit public weight downloads")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
