"""Step 3b (GPU machine, Ollama gpt-oss:20b): item profiles, what reviewers praised and criticized about each
item, from its reviews before the cutoff (item_reviews.jsonl.gz, prep_items.py). One call per item, cached like
the persona calls, so a stopped run resumes and a repeated run costs nothing.

  python item_profiles.py --input data/rr_input --out data/item_profiles --base-url http://127.0.0.1:11434/v1 --model gpt-oss:20b
  python item_profiles.py ... --items lists --run runs/<run>     # only the items in that run's lists and targets
  python item_profiles.py ... --mock                             # dry run, no LLM

The profiles do not depend on a run (only on the item's reviews), so they live in data/item_profiles/ and are
shared by every run; rerank.py reads them from there. Items already in the output file are skipped.

Outputs in <out>/:
  item_profiles.jsonl   one line per item: i, asin, praised, criticized, valid  (appended as items finish)
  calls_items.jsonl     tokens, seconds and finish reason per call of this invocation
  summary_items.json    counts, timing
"""
import argparse
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import prompts as P
from llm import LLM, DeadlineReached, parse_json
from personas import as_list


def check(obj):
    if not isinstance(obj, dict) or "praised" not in obj or "criticized" not in obj:
        return None
    praised, criticized = as_list(obj.get("praised")), as_list(obj.get("criticized"))
    if praised is None or criticized is None:
        return None
    return {"praised": praised, "criticized": criticized}


def items_of_run(run):
    """Every item index in the run's lists and targets (test and training users)."""
    ids = set()
    for name in ("candidates.jsonl", "candidates_train.jsonl"):
        f = Path(run) / name
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    u = json.loads(line)
                    ids.update(i for i, _ in u["candidates"])
                    ids.update(t["i"] for t in u["targets"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--items", default="all", choices=("all", "lists"), help="every item, or the items of --run's lists")
    ap.add_argument("--run", default=None, help="run folder with candidates.jsonl (for --items lists)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--deadline-hours", type=float, default=0, help="no new LLM calls after this (0 = none)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--cache", default=None, help="LLM answer cache folder (default <out>/cache); share it between runs")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(args.input) / "item_reviews.jsonl.gz", "rt", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.items == "lists":
        if not args.run:
            sys.exit("--items lists needs --run <run folder>")
        want = items_of_run(args.run)
        items = [x for x in items if x["i"] in want]
    out_file = out / "item_profiles.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(l)["i"] for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [x for x in items if x["i"] not in done and x["reviews"]]
    no_reviews = sum(1 for x in items if not x["reviews"])
    print(f"{len(items)} items ({args.items}), {len(done)} already profiled, {no_reviews} without reviews, "
          f"{len(todo)} to do | model {args.model} | reasoning {args.reasoning_effort} | threads {args.threads}", flush=True)
    if not todo:
        return
    llm = LLM(args.base_url, args.model, Path(args.cache) if args.cache else out / "cache", args.reasoning_effort,
              not args.no_json_mode, args.max_tokens,
              deadline=t0 + args.deadline_hours * 3600 if args.deadline_hours else None, mock=args.mock)
    lock = threading.Lock()
    fh = open(out_file, "a", encoding="utf-8")

    def job(x):
        try:
            msgs = P.item_profile(x["title"], x["reviews"])
            res = check(parse_json(llm.call(msgs, {"role": "item", "i": x["i"]})))
            if res is None:
                retry = [{"role": "user", "content": msgs[-1]["content"] + P.RETRY_NOTE_ITEM}]
                res = check(parse_json(llm.call(retry, {"role": "item", "i": x["i"], "retry": True})))
            return {"i": x["i"], "asin": x["asin"], "praised": res["praised"] if res else [],
                    "criticized": res["criticized"] if res else [], "valid": res is not None}, None
        except DeadlineReached:
            return None, "stopped (time limit)"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    n, failed, errors = 0, 0, {}
    with ThreadPoolExecutor(args.threads) as ex:
        for f in as_completed([ex.submit(job, x) for x in todo]):
            rec, err = f.result()
            if err:
                failed += 1
                errors[err] = errors.get(err, 0) + 1
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            n += 1
            if n % 200 == 0 or n == len(todo):
                s = llm.stats
                print(f"  {n}/{len(todo)} items | calls {s['calls']} (new {s['new']}) | "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    s = llm.stats
    summary = {"items_done": n, "items_failed": failed, "errors": errors, "already_done": len(done),
               "calls": s["calls"], "new_calls": s["new"], "cached_calls": s["cached"],
               "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["completion_tokens"],
               "llm_seconds": round(s["seconds"], 1), "wall_minutes": round((time.time() - t0) / 60, 1),
               "invalid": sum(1 for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip() and not json.loads(l)["valid"]),
               "params": llm.params, "model": args.model}
    (out / "calls_items.jsonl").write_text("".join(json.dumps(c) + "\n" for c in llm.log_lines), encoding="utf-8")
    (out / "summary_items.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}), flush=True)
    sys.exit(0 if n else 1)


if __name__ == "__main__":
    main()
