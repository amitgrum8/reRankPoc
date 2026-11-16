"""EXP3RT's step 3 on its IMDB release (GPU): fine-tune Llama 3.1 8B Instruct on the released reasoning data, then the
expected rating of every (user, movie) of the top-20 lists, as in Table 3. imdb.py report reads the scores.

  python exp3rt.py train --variant exact --gpu 0     # LoRA on data/imdb/raw/train.json, best validation loss kept
  python exp3rt.py train --variant clean --gpu 1     # the same without the (user, movie) pairs of the top-k test
  python exp3rt.py score --variant exact --gpu 0     # lgn and mf top-20 lists + the rating test set (RMSE / MAE)
  python exp3rt.py train --variant exact --tiny      # CPU smoke test: a 2-layer random model, 40 examples

As in the authors' code (github.com/jieyong99/EXP3RT: train.py, test.py, shell/*_imdb.sh) and the paper (Sec. 3.6, 4.1.4):
  prompt     their IMDB prompt word for word: system instructions, <User Persona> (with "[User Average Rating]"),
             <Movie Description>, <Movie Synopsis> (the item profile, with "[Item Average Rating]"); the averages and
             the target are shifted to 0-9 (rating - 1), the answer is "Reasoning: <the GPT-3.5 rationale>\\nPredicted
             User Rating: <digit>"; the loss covers the answer only; prompt + answer cut at 1,200 tokens
  training   LoRA r 512, alpha 128, dropout 0.1 on every linear layer (paper: IMDB r 512 / alpha 128; their merge script
             uses the same), AdamW lr 2e-4 constant, batch 32, gradient clipping 0.3, bf16, gradient checkpointing,
             group_by_length, at most 5 epochs, the adapter with the lowest validation loss is kept, early stop after 1
             epoch without a gain (paper: patience 1)
  rating     greedy reasoning (at most 512 new tokens), then "Predicted User Rating: " appended and the next-token
             probabilities of the digits 0-9 renormalized; expected rating = sum (d + 1) p(d) on the 1-10 scale
Differences: the base model is Llama 3.1 8B Instruct (theirs Llama 3 8B Instruct); LoRA on bf16 weights instead of
QLoRA (4-bit): the GPU has room for it, and 4-bit only saves memory (the LoRA layers are bf16 as in their code);
transformers 5 has no group_by_length (it only groups similar lengths into a batch); generation with Hugging
Face transformers instead of vLLM (vLLM does not run on Windows); the digits' probabilities come from the full
next-token distribution instead of vLLM's top-20 log-probabilities (the same up to digits outside the top 20).

Variants:
  exact   the released train.json / valid.json as they are. They hold 2,734 + 350 of the 3,426 (user, movie) pairs of
          the top-k test with their 9-10 rating: the model has seen most test pairs with the answer.
  clean   those pairs removed from train and valid (the same model and settings otherwise).
Outputs in runs/imdb/exp3rt_<variant>/: adapter/ (not pushed, about 2.7 GB), train_log.json, scores_lgn.jsonl,
scores_mf.jsonl, scores_test.jsonl (one line per pair: user_id, item_id, expected, argmax, probs, reasoning,
seconds), rating_metrics.json (RMSE / MAE on test.json), timing.json.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "imdb" / "raw"
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"     # the same weights as meta-llama/Llama-3.1-8B-Instruct, not gated
CUTOFF = 1200
DIGITS = [str(d) for d in range(10)]


def shift_avg(text, who):
    """Their imdb_process_*_rating: "[Average Rating]\\n8.3" -> "[User Average Rating]\\n7.3" (rating - 1, at least 0)."""
    parts = text.split("[Average Rating]")
    if len(parts) > 1:
        try:
            r = float(parts[-1].strip())
            return f"{parts[0]}[{who} Average Rating]\n{max(0, r - 1):.1f}"
        except ValueError:
            return text
    return text


def prompt(rec):
    """Their IMDB input text, character for character (train.py, dataset == "imdb")."""
    s_item, l_item = "movie", "Movie"
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a helpful AI assistant for {s_item} recommendation. Based on the user's preferences and {s_item} characteristics provided, generate a recommendation reasoning and predict the user's rating.\n"
        "You must always generate a response in the following format whenever the user provides information:\n"
        f"Reasoning: [Provide a detailed, single-paragraph reasoning for your prediction, addressing at least three specific points of alignment or misalignment between the user's preferences and the {s_item}'s characteristics.]\n"
        f"Predicted User Rating: [Predict the user's rating as an integer from 0 to 9: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. 0 indicates the user would strongly dislike the {s_item}, while 9 indicates the user would highly enjoy and recommend it. Consider the average ratings provided for the user and the {s_item} in your prediction.]\n"
        "Note: Do not simply repeat the input text. Generate a new reasoning and rating prediction based on the input provided.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"I need a recommendation for this {s_item}. Here's the information:\n"
        "User Preferences:\n"
        "<User Persona>\n"
        f"{shift_avg(rec['user_persona'], 'User')}\n\n"
        f"{l_item} Characteristics:\n"
        f"<{l_item} Description>\n"
        f"{rec['item_description']}\n\n"
        f"<{l_item} Synopsis>\n"
        f"{shift_avg(rec['item_synopsis'], 'Item')}\n\n"
        f"Based on this information, please provide a detailed reasoning for your recommendation and predict a rating for this {s_item}. Follow the format specified in the system instructions.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )


def answer(rec):
    return f"Reasoning: {rec['rationale']}\nPredicted User Rating: {int(rec['score']) - 1}<|eot_id|><|end_of_text|>"


def load(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def test_pairs():
    """The (user, movie) pairs of the top-k test (topk_test.txt), as (EXP3RT user id, int movie id)."""
    rows = lambda f: [l.split() for l in (RAW / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    uid = {int(r[1]): r[0] for r in rows("topk_user_list.txt")[1:]}
    iid = {int(r[1]): int(r[0]) for r in rows("topk_item_list.txt")[1:]}
    return {(uid[int(r[0])], iid[int(x)]) for r in rows("topk_test.txt") for x in r[1:]}


def records(split, variant):
    recs = load(f"{split}.json")
    if variant == "clean":
        drop = test_pairs()
        recs = [r for r in recs if (r["user_id"], int(r["item_id"])) not in drop]
    return recs


def out_dir(variant, tiny):
    return HERE / "runs" / "imdb" / (f"exp3rt_{variant}" + ("_tiny" if tiny else ""))


def tokenizer_of(base):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    pad = "<|finetune_right_pad_id|>"                  # Llama 3.1's padding token
    tok.pad_token = pad if pad in tok.get_vocab() else tok.eos_token
    return tok


def tiny_model(tok):
    """A 2-layer random Llama with the real tokenizer, for CPU smoke tests of the code path."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    return LlamaForCausalLM(cfg)


def load_model(base, tiny, device):
    import torch
    from transformers import AutoModelForCausalLM
    tok = tokenizer_of(base)
    if tiny:
        return tiny_model(tok).to(device), tok
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": device},
                                                 attn_implementation="sdpa")
    return model, tok


# ---------------------------------------------------------------------------------------------------- train
def train(args):
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    out = out_dir(args.variant, args.tiny)
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.tiny else f"cuda:{args.gpu}"
    torch.manual_seed(args.seed)
    tr, va = records("train", args.variant), records("valid", args.variant)
    if args.tiny:
        tr, va = tr[:40], va[:8]
    model, tok = load_model(args.base_model, args.tiny, device)

    def encode(rec):
        inp = prompt(rec)
        full = tok(f"{inp}\n{answer(rec)}", truncation=True, max_length=CUTOFF, add_special_tokens=False)
        n_in = len(tok(inp, truncation=True, max_length=CUTOFF, add_special_tokens=False)["input_ids"])
        full["labels"] = [-100] * n_in + full["input_ids"][n_in:]
        return full

    t0 = time.time()
    data_tr, data_va = [encode(r) for r in tr], [encode(r) for r in va]
    cut = sum(len(x["input_ids"]) >= CUTOFF for x in data_tr)
    print(f"{args.variant}: {len(data_tr)} training and {len(data_va)} validation examples ({cut} cut at {CUTOFF} "
          f"tokens) | tokenized in {time.time() - t0:.0f}s", flush=True)
    r, alpha = (8, 16) if args.tiny else (args.lora_r, args.lora_alpha)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    if not args.tiny:                                 # as in their train.py: the LoRA layers in bf16 (half the memory)
        from peft.tuners.lora import LoraLayer
        for m in model.modules():
            if isinstance(m, LoraLayer):
                m.to(torch.bfloat16)
    model.print_trainable_parameters()
    if not args.tiny:
        model.enable_input_require_grads()            # gradient checkpointing with frozen embeddings
    model.config.use_cache = False

    class KeepBest(transformers.TrainerCallback):
        """Saves the adapter whenever the validation loss improves; stops after --patience epochs without a gain."""
        def __init__(self):
            self.best, self.bad, self.history = float("inf"), 0, []

        def on_evaluate(self, a, state, control, metrics=None, **kw):
            loss = (metrics or {}).get("eval_loss")
            if loss is None:
                return control
            self.history.append({"epoch": round(state.epoch, 2), "eval_loss": round(loss, 5),
                                 "minutes": round((time.time() - t0) / 60, 1)})
            print(f"epoch {state.epoch:.2f}: validation loss {loss:.5f} (best {self.best:.5f})", flush=True)
            if loss < self.best:
                self.best, self.bad = loss, 0
                model.save_pretrained(out / "adapter")
                tok.save_pretrained(out / "adapter")
            else:
                self.bad += 1
                if self.bad >= args.patience:
                    control.should_training_stop = True
            return control

    keep = KeepBest()
    micro = 1 if args.tiny else args.micro_batch
    targs = dict(output_dir=str(out / "trainer"), per_device_train_batch_size=micro, per_device_eval_batch_size=micro,
                 gradient_accumulation_steps=max(1, args.batch // micro) if not args.tiny else 2,
                 num_train_epochs=1 if args.tiny else args.epochs, learning_rate=args.lr, lr_scheduler_type="constant",
                 weight_decay=0.0, max_grad_norm=0.3, bf16=not args.tiny, logging_steps=10,
                 eval_strategy="epoch", save_strategy="no", group_by_length=True,
                 gradient_checkpointing=not args.tiny, report_to=[], seed=args.seed, remove_unused_columns=False,
                 dataloader_num_workers=0)
    # (their warmup_ratio 0.03 does nothing under the "constant" scheduler, and transformers 5 dropped it)
    import inspect
    known = set(inspect.signature(transformers.TrainingArguments.__init__).parameters)
    dropped = sorted(set(targs) - known)
    if dropped:
        print(f"this transformers ({transformers.__version__}) has no {', '.join(dropped)}: left out", flush=True)
    targs = {k: v for k, v in targs.items() if k in known}
    trainer = transformers.Trainer(
        model=model, args=transformers.TrainingArguments(**targs), train_dataset=data_tr, eval_dataset=data_va,
        data_collator=transformers.DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[keep])
    t1 = time.time()
    trainer.train()
    log = {"variant": args.variant, "base_model": args.base_model, "tiny": args.tiny, "train_examples": len(data_tr),
           "valid_examples": len(data_va), "cut_at_cutoff": cut, "lora_r": r, "lora_alpha": alpha, "lr": args.lr,
           "batch": args.batch, "micro_batch": micro, "best_eval_loss": keep.best, "epochs": keep.history,
           "train_hours": round((time.time() - t1) / 3600, 3),
           "train_loss": [x for x in trainer.state.log_history if "loss" in x][-20:]}
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"done: best validation loss {keep.best:.5f}, {log['train_hours']:.2f} h -> {out / 'adapter'}", flush=True)


# ---------------------------------------------------------------------------------------------------- score
def parse_reasoning(text):
    """The reasoning paragraph of a generated answer (their test.py: the text after "Reasoning:", up to the rating)."""
    t = text.split("Reasoning:", 1)[1] if "Reasoning:" in text else text
    t = t.split("Predicted User Rating", 1)[0]
    t = t.replace("<|eot_id|>", "").strip()
    return t or "No reasoning provided."


def score(args):
    import numpy as np
    import torch
    from peft import PeftModel

    out = out_dir(args.variant, args.tiny)
    device = "cpu" if args.tiny else f"cuda:{args.gpu}"
    if not (out / "adapter").exists():
        sys.exit(f"{out / 'adapter'} is missing: train first")
    model, tok = load_model(args.base_model, args.tiny, device)
    model = PeftModel.from_pretrained(model, out / "adapter").merge_and_unload()
    model.eval()
    tok.padding_side = "left"
    digit_ids = [tok.convert_tokens_to_ids(d) for d in DIGITS]
    probe = "Predicted User Rating: "
    a, b = tok(probe, add_special_tokens=False)["input_ids"], tok(probe + "7", add_special_tokens=False)["input_ids"]
    assert b[:len(a)] == a and b[len(a):] == [digit_ids[7]], "the digit after the rating prefix is not one token"
    stop = [tok.convert_tokens_to_ids("<|eot_id|>"), tok.eos_token_id]
    timing = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}

    for name in args.lists:
        recs = load({"lgn": "lgn_test_top20.json", "mf": "mf_test_top20.json", "test": "test.json"}[name])
        if args.tiny:
            recs = recs[:6]
        path = out / f"scores_{name}.jsonl"
        done = set()
        if path.exists():
            done = {(x["user_id"], x["item_id"]) for x in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                    if x}
        todo = [r for r in recs if (r["user_id"], r["item_id"]) not in done]
        print(f"{name}: {len(recs)} pairs, {len(done)} already scored, {len(todo)} to do", flush=True)
        todo.sort(key=lambda r: len(prompt(r)))         # similar lengths in a batch: less padding
        t0, n = time.time(), 0
        with open(path, "a", encoding="utf-8") as f:
            for k in range(0, len(todo), args.batch_size):
                batch = todo[k:k + args.batch_size]
                t1 = time.time()
                prompts = [prompt(r) for r in batch]
                enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=8 if args.tiny else 512, do_sample=False,
                                         eos_token_id=stop, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
                reasons = [parse_reasoning(t) for t in texts]
                second = [f"{p}\nReasoning: {r}\nPredicted User Rating: " for p, r in zip(prompts, reasons)]
                enc2 = tok(second, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                with torch.no_grad():
                    logits = model(**enc2).logits[:, -1, :].float()
                lp = torch.log_softmax(logits, -1)[:, digit_ids].cpu().numpy()
                sec = (time.time() - t1) / len(batch)
                for r, row, reason in zip(batch, lp, reasons):
                    p = np.exp(row - row.max())
                    p /= p.sum()
                    f.write(json.dumps({"user_id": r["user_id"], "item_id": r["item_id"],
                                        "expected": round(float(sum((d + 1) * p[d] for d in range(10))), 4),
                                        "argmax": int(p.argmax()) + 1, "probs": [round(float(x), 4) for x in p],
                                        "digit_mass": round(float(np.exp(row).sum()), 4),
                                        "score": r.get("score"), "reasoning": reason, "seconds": round(sec, 3)},
                                       ensure_ascii=False) + "\n")
                f.flush()
                n += len(batch)
                if (k // args.batch_size) % 10 == 0:
                    rate = n / (time.time() - t0)
                    print(f"  {name}: {len(done) + n}/{len(recs)} pairs, {rate:.2f} pairs/s, "
                          f"{(len(todo) - n) / max(rate, 1e-9) / 60:.0f} min left", flush=True)
        if n:
            timing[f"{name}_pairs"] = len(recs)
            timing[f"{name}_seconds_per_pair"] = round((time.time() - t0) / n, 3)
            timing[f"{name}_seconds_per_user_of_20"] = round(20 * (time.time() - t0) / n, 1)
        if name == "test":
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            y = np.array([float(x["score"]) for x in rows])
            e, m = np.array([x["expected"] for x in rows]), np.array([x["argmax"] for x in rows], dtype=float)
            met = {"pairs": len(rows), "rmse_expected": round(float(np.sqrt(((y - e) ** 2).mean())), 4),
                   "mae_expected": round(float(np.abs(y - e).mean()), 4),
                   "rmse_argmax": round(float(np.sqrt(((y - m) ** 2).mean())), 4),
                   "mae_argmax": round(float(np.abs(y - m).mean()), 4)}
            (out / "rating_metrics.json").write_text(json.dumps(met, indent=1), encoding="utf-8")
            print("rating test: " + json.dumps(met), flush=True)
    timing["gpu"] = "cpu (tiny)" if args.tiny else torch.cuda.get_device_name(args.gpu)
    (out / "timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("train", "score"))
    ap.add_argument("--variant", default="exact", choices=("exact", "clean"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test with a 2-layer random model")
    ap.add_argument("--lora-r", type=int, default=512)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=4, help="their micro_batch_size; 2 or 1 if the GPU runs out of memory")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lists", nargs="+", default=["lgn", "mf", "test"], choices=("lgn", "mf", "test"))
    ap.add_argument("--batch-size", type=int, default=16, help="score: pairs generated at once")
    args = ap.parse_args()
    if not (RAW / "fetch.json").exists():
        sys.exit("data/imdb/raw is empty: run  python imdb.py fetch")
    train(args) if args.cmd == "train" else score(args)


if __name__ == "__main__":
    main()
