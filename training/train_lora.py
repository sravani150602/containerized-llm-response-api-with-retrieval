"""Fine-tune an open-source chat LLM with LoRA on workplace conversations.

LoRA in one paragraph
---------------------
Instead of updating all ~360M weights of the model, LoRA freezes them and
learns a pair of small matrices A (d x r) and B (r x d) for selected layers,
so each adapted weight becomes W + (alpha / r) * B @ A. With r=16 that is
well under 1% of the parameters. It trains on a laptop CPU, the adapter is a
few MB, and it can be merged back into W for zero-overhead inference.

Only the assistant's answer tokens contribute to the loss (the system prompt,
context and question are masked with -100), so the model learns *how to
answer*, not to reproduce the prompt.

Usage:
    python training/train_lora.py                      # defaults below
    python training/train_lora.py --epochs 1 --max-train 400   # quicker run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def encode(tok, messages: list[dict], max_len: int) -> dict | None:
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(f_ids) > max_len:
        return None  # skip over-long examples instead of truncating the answer
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels}


def batches(examples: list[dict], bs: int, pad_id: int, shuffle: bool, seed: int):
    idx = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [examples[j] for j in idx[i:i + bs]]
        n = max(len(e["input_ids"]) for e in chunk)
        ids = torch.full((len(chunk), n), pad_id)
        lab = torch.full((len(chunk), n), -100)
        att = torch.zeros((len(chunk), n), dtype=torch.long)
        for k, e in enumerate(chunk):
            L = len(e["input_ids"])
            ids[k, :L] = torch.tensor(e["input_ids"])
            lab[k, :L] = torch.tensor(e["labels"])
            att[k, :L] = 1
        yield {"input_ids": ids, "labels": lab, "attention_mask": att}


@torch.no_grad()
def evaluate(model, data, bs, pad_id) -> float:
    model.eval()
    total, count = 0.0, 0
    for b in batches(data, bs, pad_id, False, 0):
        out = model(**b)
        n = (b["labels"] != -100).sum().item()
        total += out.loss.item() * n
        count += n
    model.train()
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--data", default="training/data")
    ap.add_argument("--out", default="artifacts/lora-adapter")
    ap.add_argument("--log", default="results/training_log.json")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)

    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    data_dir = Path(args.data)
    train = [e for r in load_jsonl(data_dir / "train.jsonl", args.max_train)
             if (e := encode(tok, r["messages"], args.max_len))]
    val = [e for r in load_jsonl(data_dir / "val.jsonl", 60) if (e := encode(tok, r["messages"], args.max_len))]
    print(f"train examples: {len(train)}  val examples: {len(val)}", flush=True)

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, total_steps // 20), total_steps)

    log = {"args": vars(args), "trainable_params": trainable, "total_params": total, "train_examples": len(train),
           "steps": [], "val": []}
    log["val"].append({"step": 0, "loss": round(evaluate(model, val, args.batch_size, pad_id), 4)})
    print(f"step 0 val_loss {log['val'][-1]['loss']}", flush=True)

    model.train()
    step, micro, t0, epoch = 0, 0, time.time(), 0
    running = []
    while step < total_steps:
        for b in batches(train, args.batch_size, pad_id, True, args.seed + epoch):
            loss = model(**b).loss / args.grad_accum
            loss.backward()
            running.append(loss.item() * args.grad_accum)
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            entry = {"step": step, "loss": round(sum(running) / len(running), 4),
                     "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
            log["steps"].append(entry)
            running = []
            if step % 10 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} loss {entry['loss']} elapsed {entry['elapsed_s']}s", flush=True)
            if step % max(1, total_steps // 4) == 0 or step == total_steps:
                v = round(evaluate(model, val, args.batch_size, pad_id), 4)
                log["val"].append({"step": step, "loss": v})
                print(f"step {step} val_loss {v}", flush=True)
                Path(args.log).parent.mkdir(parents=True, exist_ok=True)
                Path(args.log).write_text(json.dumps(log, indent=2))
            if step >= total_steps:
                break
        epoch += 1

    log["train_seconds"] = round(time.time() - t0, 1)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    Path(args.log).write_text(json.dumps(log, indent=2))
    print(f"saved adapter to {args.out} in {log['train_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
