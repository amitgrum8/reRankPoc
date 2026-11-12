"""Step 3 (GPU machine, Ollama gpt-oss:20b): EXP3RT personas for the test users, and for the reranker's
training users (--set train: reads train_users.json, writes personas_train.jsonl, extractions_train.jsonl, ...).

  python personas.py --input data/rr_input --out run --base-url http://127.0.0.1:11434/v1 --model gpt-oss:20b
  python personas.py --input data/rr_input --out run --set train --base-url ... --model ...
  python personas.py --input data/rr_input --out run_mock --mock          # dry run, no LLM

Per test user: EXP3RT step 1 on each history review (the user's most recent reviews before the cutoff),
then step 2 on all the step-1 outputs -> {"like": [...], "dislike": [...]}, about 5 each.
An answer that is not valid JSON is asked once more with a note; if it still fails, the extraction counts
as empty (and is flagged), or, for step 2, the user fails.
Every call is cached in <out>/cache/, so a stopped run resumes where it stopped.

Outputs in <out>/:
  extractions.jsonl  one line per history review: user, line_no, item, stars, like, dislike, valid
  personas.jsonl     one line per user: user_id, u, like, dislike, n_reviews, n_invalid_extractions
  calls.jsonl        tokens, seconds and finish reason per call
  summary.json       counts, failures, timing
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import prompts as P
from llm import LLM, DeadlineReached, parse_json


def as_list(x):
    """A list of non-empty strings, or None if x is not a list. Accepts items like {"text": ...}."""
    if not isinstance(x, list):
        return None
    out = []
    for v in x:
        if isinstance(v, dict):
            v = v.get("text") or v.get("preference") or ""
        if isinstance(v, str) and v.strip() and v.strip().lower() not in ("none", "none."):
            out.append(v.strip())
    return out


def check(obj, both_tags=False):
    """both_tags: the answer must contain "like" and "dislike" (the profile step; the model used to leave the
    dislike tag out for a quarter of the users, which read as a persona with no dislikes)."""
    if not isinstance(obj, dict):
        return None
    if both_tags and not ("like" in obj and "dislike" in obj):
        return None
    like, dislike = as_list(obj.get("like", [])), as_list(obj.get("dislike", []))
    if like is None or dislike is None:
        return None
    return {"like": like, "dislike": dislike}


def ask(llm, messages, meta, both_tags=False):
    """One call, validated; one retry with a note. Returns (result or None, missing_tag): after the retry an
    answer with a missing tag is accepted and flagged (both_tags), so a user is not lost over it."""
    res = check(parse_json(llm.call(messages, meta)), both_tags)
    if res is not None:
        return res, False
    retry = [{"role": "user", "content": messages[-1]["content"] + P.RETRY_NOTE}]
    ans = parse_json(llm.call(retry, {**meta, "retry": True}))
    res = check(ans, both_tags)
    if res is None and both_tags:
        return check(ans), True
    return res, False


def run_user(llm, tu):
    ex_rows = []
    for k, h in enumerate(tu["history"]):
        meta = {"role": "extract", "user": tu["user_id"], "review": k}
        res, _ = ask(llm, P.extract(h.get("review_title", ""), h.get("text", "")), meta)
        ex_rows.append({"user_id": tu["user_id"], "k": k, "line_no": h["line_no"], "asin": h["asin"], "i": h["i"],
                        "title": h["title"], "rating": h["rating"], "ts": h["ts"],
                        "like": res["like"] if res else [], "dislike": res["dislike"] if res else [],
                        "valid": res is not None})
    persona, missing = ask(llm, P.profile(ex_rows), {"role": "profile", "user": tu["user_id"]}, both_tags=True)
    if persona is None:
        raise ValueError("profile answer not valid JSON after a retry")
    return ex_rows, {"user_id": tu["user_id"], "u": tu["u"], "like": persona["like"], "dislike": persona["dislike"],
                     "n_reviews": len(ex_rows), "n_invalid_extractions": sum(not r["valid"] for r in ex_rows),
                     "missing_tag": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--set", default="test", choices=("test", "train"),
                    help="test_users.json -> personas.jsonl etc.; train_users.json (the reranker's training users) "
                         "-> personas_train.jsonl etc.")
    ap.add_argument("--users", default="all", help="'all' or the number of users of the set to run (first n)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--deadline-hours", type=float, default=0, help="no new LLM calls after this (0 = none)")
    ap.add_argument("--start-deadline-hours", type=float, default=0, help="no new users after this (0 = none)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--cache", default=None, help="LLM answer cache folder (default <out>/cache); share it between runs")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    users = json.loads((Path(args.input) / f"{args.set}_users.json").read_text(encoding="utf-8"))["users"]
    sfx = "" if args.set == "test" else "_train"
    if args.users != "all":
        users = users[:int(args.users)]
    llm = LLM(args.base_url, args.model, Path(args.cache) if args.cache else out / "cache", args.reasoning_effort, not args.no_json_mode,
              args.max_tokens, deadline=t0 + args.deadline_hours * 3600 if args.deadline_hours else None,
              mock=args.mock)
    start_deadline = t0 + args.start_deadline_hours * 3600 if args.start_deadline_hours else None
    n_calls = sum(len(tu["history"]) + 1 for tu in users)
    print(f"{len(users)} {args.set} users, about {n_calls} calls | model {args.model} | reasoning {args.reasoning_effort} | "
          f"threads {args.threads}", flush=True)

    done, failed, extractions, personas = 0, {}, [], []

    def job(tu):
        if start_deadline and time.time() > start_deadline:
            return tu, None, "not started (time limit)"
        try:
            return tu, run_user(llm, tu), None
        except DeadlineReached:
            return tu, None, "stopped (time limit)"
        except Exception as e:
            return tu, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(args.threads) as ex:
        futures = [ex.submit(job, tu) for tu in users]
        for f in as_completed(futures):
            tu, res, err = f.result()
            if err:
                failed[tu["user_id"]] = err
                print(f"user {tu['user_id']} FAILED: {err}", flush=True)
                continue
            ex_rows, persona = res
            extractions += ex_rows
            personas.append(persona)
            done += 1
            s = llm.stats
            print(f"user {tu['user_id']} done ({done}/{len(users)}) | {len(persona['like'])} likes, "
                  f"{len(persona['dislike'])} dislikes | calls {s['calls']} (new {s['new']}) | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    order = {tu["user_id"]: k for k, tu in enumerate(users)}
    extractions.sort(key=lambda r: (order[r["user_id"]], r["k"]))
    personas.sort(key=lambda r: order[r["user_id"]])
    write = lambda name, rows: (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                       encoding="utf-8")
    write(f"extractions{sfx}.jsonl", extractions)
    write(f"personas{sfx}.jsonl", personas)
    write(f"calls{sfx}.jsonl", llm.log_lines)
    s = llm.stats
    new_calls = [c for c in llm.log_lines if not c["cached"]]
    summary = {"users_done": done, "users_failed": failed, "calls": s["calls"], "new_calls": s["new"],
               "cached_calls": s["cached"], "prompt_tokens": s["prompt_tokens"],
               "completion_tokens": s["completion_tokens"], "reasoning_tokens": s["reasoning_tokens"],
               "llm_seconds": round(s["seconds"], 1), "wall_minutes": round((time.time() - t0) / 60, 1),
               "finish_length": sum(c.get("finish_reason") == "length" for c in new_calls),
               "invalid_extractions": sum(not r["valid"] for r in extractions),
               "profiles_missing_tag": sum(bool(p.get("missing_tag")) for p in personas),
               "profiles_no_dislikes": sum(not p["dislike"] for p in personas),
               "retries": sum(bool(c.get("retry")) for c in llm.log_lines),
               "params": llm.params, "model": args.model}
    (out / f"summary{sfx}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}), flush=True)
    sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()
