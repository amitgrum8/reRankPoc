"""GPU machine, one command: R13_near (training users weighted by persona closeness), the ablations A1-A6 of the
dislike part and the standard metrics (Recall/NDCG with paired tests, per-target ranks) on the three base rankers,
with Qwen3-Embedding-8B and alpha = beta = 0.1 as in eval_qwen8b.md; then commits and pushes the results.

  git pull
  uv run python run_r13.py                 # smoke pass (10 test / 40 training users), then the three full runs
  uv run python run_r13.py --no-push       # the same, results stay local
  uv run python run_r13.py --no-smoke --runs runs\\2026-11-14_sasrec

No LLM: the personas are the ones already in each run folder. Per run it writes eval_<tag>.md, mix_<tag>.json,
ranks_<tag>.jsonl.gz and the console output evaluate_<tag>.txt (tag qwen8b_r13 by default). R7_lgbm is skipped,
so the committed lgbm.txt / lgbm.json stay as they are. The movie and review vectors come from the Qwen caches in
data/vectors/ (made by the earlier runs); without them the first run re-encodes every review sentence (hours).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = ["runs/2026-11-14_1057", "runs/2026-11-14_pop", "runs/2026-11-14_sasrec"]    # LightGCN, popularity, SASRec
NEEDED = ["candidates.jsonl", "candidates_train.jsonl", "personas.jsonl", "personas_train.jsonl"]
OUTPUTS = ["eval_{tag}.md", "mix_{tag}.json", "ranks_{tag}.jsonl.gz", "evaluate_{tag}.txt"]


def say(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tee(cmd, log_path):
    """Runs cmd, printing its output and writing it to log_path; returns the exit code."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(" ".join(str(c) for c in cmd) + "\n\n")
        p = subprocess.Popen([str(c) for c in cmd], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        for raw in p.stdout:
            line = raw.decode("utf-8", errors="replace")
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return p.wait()


def git(*args, check=False):
    r = subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{out}")
    return r.returncode, out


def evaluate_cmd(run, args, tag, extra=()):
    return [sys.executable, "-u", HERE / "evaluate.py", run, "--encoder", args.encoder, "--alpha", args.alpha,
            "--beta", args.beta, "--tag", tag, "--no-lgbm", "--show", "0", *extra]


def highlights(eval_md):
    """The lines worth seeing at once: R13's alpha, and the comparisons with R12 and the swapped persona."""
    keep = ("| R13_near - ", "| R13_near_ctrl - ", "| A1_likes_only - ", "| A2_dis_all - ", "| A3_dis_good - ",
            "| A4_mixed - ", "| A5_coral - ", "| A6_satisfaction - ", "| R12_mix_reviews: own", "| R13_near: own")
    out = []
    for line in eval_md.read_text(encoding="utf-8").splitlines():
        if line.startswith(keep) or "alpha picked" in line:
            out.append(line if len(line) < 400 else line[:400] + " ...")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=RUNS, help="run folders (default: the three base rankers)")
    ap.add_argument("--encoder", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--alpha", default="0.1", help="formula weight of the like side, as in eval_qwen8b.md")
    ap.add_argument("--beta", default="0.1", help="formula weight of the dislike side, as in eval_qwen8b.md")
    ap.add_argument("--tag", default="qwen8b_r13")
    ap.add_argument("--no-smoke", action="store_true", help="skip the 10-user smoke pass")
    ap.add_argument("--no-push", action="store_true", help="commit nothing, push nothing")
    ap.add_argument("--allow-cpu", action="store_true", help="run even without a GPU (Qwen-8B on a CPU is very slow)")
    args = ap.parse_args()
    t0 = time.time()

    say("checks")
    _, head = git("log", "-1", "--format=%h %s")
    print(f"code at commit {head}")
    try:
        import torch
        gpu = torch.cuda.is_available()
        print(f"torch {torch.__version__}, GPU: " + (torch.cuda.get_device_name(0) if gpu else "none"))
    except ImportError:
        gpu = False
        print("torch is not installed in this environment (run with: uv run python run_r13.py)")
    if not gpu and not args.allow_cpu:
        sys.exit("no GPU visible: stopping (--allow-cpu to run anyway)")
    missing = [f"{r}/{f}" for r in args.runs for f in NEEDED if not (HERE / r / f).exists()]
    missing += [str(p) for p in (HERE / "data" / "rr_input" / "item_reviews.jsonl.gz",
                                 HERE / "data" / "item_profiles" / "item_profiles.jsonl") if not p.exists()]
    if missing:
        sys.exit("missing files (git pull?):\n  " + "\n  ".join(missing))
    slug = args.encoder.replace("/", "__")
    caches = sorted(p.name for p in (HERE / "data" / "vectors").glob(slug + "*.npz")) if (HERE / "data" / "vectors").exists() else []
    print("vector caches: " + (", ".join(caches) if caches else "NONE - the first run encodes every movie and review "
                                                                  "sentence again (hours)"))

    if not args.no_smoke:
        run = args.runs[0]
        say(f"smoke pass on {run}: 10 test users, 40 training users")
        log = HERE / run / "evaluate_smoke.txt"
        code = tee(evaluate_cmd(run, args, "smoke", ["--users", "10", "--train-users", "40"]), log)
        if code:
            sys.exit(f"smoke pass failed (exit {code}); the output is in {log}. Nothing committed.")
        for f in OUTPUTS:
            (HERE / run / f.format(tag="smoke")).unlink(missing_ok=True)
        print("smoke pass ok")

    done, failed = [], []
    for k, run in enumerate(args.runs, 1):
        say(f"run {k}/{len(args.runs)}: {run}")
        t1 = time.time()
        code = tee(evaluate_cmd(run, args, args.tag), HERE / run / f"evaluate_{args.tag}.txt")
        if code:
            failed.append(run)
            print(f"FAILED (exit {code}) after {(time.time() - t1) / 60:.1f} min; see {run}/evaluate_{args.tag}.txt")
        else:
            done.append(run)
            print(f"ok, {(time.time() - t1) / 60:.1f} min")

    say(f"summary ({(time.time() - t0) / 60:.0f} min)")
    for run in done:
        print(f"\n{run}:")
        for line in highlights(HERE / run / f"eval_{args.tag}.md"):
            print("  " + line)
    if failed:
        print("\nfailed: " + ", ".join(failed))

    files = [f"{run}/{f.format(tag=args.tag)}" for run in done + failed for f in OUTPUTS
             if (HERE / run / f.format(tag=args.tag)).exists()]
    if args.no_push or not files:
        print("\nnot committed" + (" (--no-push)" if args.no_push else " (no result files)") + ". Files:\n  " + "\n  ".join(files))
        return
    say("commit and push")
    git("add", "-f", *files, check=True)
    msg = (f"Results {args.tag}: R13_near, ablations A1-A6, standard metrics ({len(done)} of {len(args.runs)} runs)\n\n"
           + "\n".join(f"- {r}" + (" (FAILED, log only)" if r in failed else "") for r in args.runs))
    code, out = git("commit", "-m", msg)
    print(out)
    if code:
        print("nothing committed")
        return
    code, out = git("push")
    if code:                                              # someone pushed meanwhile: put this commit on top
        print(out)
        git("pull", "--rebase", "--autostash")
        code, out = git("push")
    print(out)
    print("pushed" if code == 0 else "PUSH FAILED: the commit is local; push it by hand with: git push")


if __name__ == "__main__":
    main()
