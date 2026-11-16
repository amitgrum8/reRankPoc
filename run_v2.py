"""GPU machine, one command (2026-11-16): EXP3RT's IMDB test (our methods, then EXP3RT itself fine-tuned), then the
larger Movies & TV set with Llama 3.1 8B personas and Qwen3-Embedding-8B, on three base rankers. After each stage the
results are committed and pushed.

  git pull
  uv run python run_v2.py                   # every stage (uv installs peft and accelerate the first time)
  uv run python run_v2.py --stages imdb     # only some stages: imdb exp3rt smoke lgcn bases eval
  uv run python run_v2.py --no-push         # results stay local
A finished stage is skipped when the runner is started again.

Stages:
  imdb    EXP3RT's IMDB release (imdb.py fetch), then imdb.py run with Qwen3-Embedding-8B and 3 seeds: the persona
          inside LightGCN / BPR-MF (ablation of every input) and our reranker on EXP3RT's top-20 lists
          -> runs/imdb/eval_imdb.md (minutes)
  exp3rt  EXP3RT's student: Llama 3.1 8B Instruct (16 GB download, once) fine-tuned on the released reasoning data,
          variant "exact" on GPU 0 and "clean" (without the top-k test pairs) on GPU 1 at the same time (hours each),
          then both score the top-20 lists and the rating test set; imdb.py report puts EXP3RT into eval_imdb.md.
          The adapters (2.7 GB each) stay on this machine. A failure here is reported and the next stages still run.
  smoke   run.py --smoke on data/rr_input_v2 into a scratch folder: proves Ollama + llama3.1:8b-instruct-fp16 (pulled
          once, 16 GB) and the larger data end to end; the folder is deleted afterwards
  lgcn    run.py on data/rr_input_v2 -> runs/v2_lgcn: LightGCN (up to 2 h on 3.2M interactions), then Llama 3.1 8B
          personas for the 1,000 test and 854 training users (about 20,000 calls, many hours), then a first
          evaluation with bge-small (eval.md). No LLM item profiles (R12 uses the reviews themselves).
  bases   popularity and SASRec (up to 2 h) on data/rr_input_v2 -> runs/v2_pop, runs/v2_sasrec, with the personas of
          runs/v2_lgcn copied in
  eval    evaluate.py with Qwen3-Embedding-8B, alpha = beta = 0.1 on the three runs -> eval_qwen8b.md each (the
          review sentences of every movie in the lists are encoded once, hours)
Not pushed: embeddings.npz (140 MB for 547k users and movies) and ollama.log.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = "data/rr_input_v2"
RUNS = {"lgcn": "runs/v2_lgcn", "pop": "runs/v2_pop", "sasrec": "runs/v2_sasrec"}
ENCODER = "Qwen/Qwen3-Embedding-8B"
STAGES = ("imdb", "exp3rt", "smoke", "lgcn", "bases", "eval")
SKIP_PUSH = {"embeddings.npz", "ollama.log"}
SKIP_DIRS = {"cache", "adapter", "trainer"}           # LLM caches, LoRA adapters (GBs), trainer scratch


def say(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tee(cmd, log_path):
    """Runs cmd, printing its output and writing it to log_path; returns the exit code."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
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


def push(folders, message, args):
    """Commits every file of the folders (except SKIP_PUSH) and pushes; a failed push leaves the commit local."""
    if args.no_push:
        print("not pushed (--no-push)")
        return
    files = [str(p.relative_to(HERE)) for f in folders for p in sorted((HERE / f).rglob("*"))
             if p.is_file() and p.name not in SKIP_PUSH and not SKIP_DIRS & set(p.relative_to(HERE).parts)]
    if not files:
        print("nothing to push")
        return
    git("add", "-f", *files, check=True)
    code, out = git("commit", "-m", message)
    print(out[-2000:])
    if code:
        print("nothing committed")
        return
    code, out = git("push")
    if code:
        print(out)
        git("pull", "--rebase", "--autostash")
        code, out = git("push")
    print("pushed" if code == 0 else f"PUSH FAILED (the commit is local; push it by hand: git push)\n{out}")


def py(*a):
    return [sys.executable, "-u", *a]


def parallel(jobs):
    """Runs (cmd, log_path) jobs at the same time, each writing only to its log; returns the exit codes."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    procs = []
    for cmd, log in jobs:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        print(f"$ {' '.join(str(c) for c in cmd)}\n  -> {log}", flush=True)
        fh = open(log, "a", encoding="utf-8")
        procs.append((subprocess.Popen([str(c) for c in cmd], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=env), fh, log))
    codes = []
    for p, fh, log in procs:
        codes.append(p.wait())
        fh.close()
        tail = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
        print(f"{log}: exit {codes[-1]}\n  " + "\n  ".join(tail), flush=True)
    return codes


def exp3rt_stage(args, n_gpus):
    out = HERE / "runs" / "imdb"
    variants = ("exact", "clean")
    if not (HERE / "data" / "imdb" / "raw" / "fetch.json").exists():
        if tee(py("imdb.py", "fetch"), out / "imdb_log.txt"):
            return False
    say("exp3rt: the base model (once)")
    import exp3rt as X
    code = tee(py("-c", f"from huggingface_hub import snapshot_download as s; print(s({X.BASE_MODEL!r}, "
                         "allow_patterns=['*.json', '*.safetensors']))"), out / "exp3rt_download.txt")
    if code:
        return False
    gpu = lambda k: str(k % max(n_gpus, 1))
    todo = [v for v in variants if not (out / f"exp3rt_{v}" / "train_log.json").exists()]
    say(f"exp3rt: training {', '.join(todo) or 'nothing (done)'} on {min(len(todo), n_gpus)} GPU(s)")
    if todo:
        batches = [todo] if n_gpus >= len(todo) else [[v] for v in todo]
        for batch in batches:
            codes = parallel([(py("exp3rt.py", "train", "--variant", v, "--gpu", gpu(variants.index(v))),
                               out / f"exp3rt_{v}" / "train.txt") for v in batch])
            if any(codes):
                return False
    say("exp3rt: scoring the top-20 lists and the rating test set")
    batches = [list(variants)] if n_gpus >= 2 else [[v] for v in variants]
    for batch in batches:
        if any(parallel([(py("exp3rt.py", "score", "--variant", v, "--gpu", gpu(variants.index(v))),
                          out / f"exp3rt_{v}" / "score.txt") for v in batch])):
            return False
    return tee(py("imdb.py", "report", "--out", out), out / "imdb_log.txt") == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--seeds", type=int, default=3, help="imdb: training runs per backbone variant")
    args = ap.parse_args()
    t0 = time.time()

    say("checks")
    print(f"code at commit {git('log', '-1', '--format=%h %s')[1]}")
    try:
        import torch
        gpu = torch.cuda.is_available()
        print(f"torch {torch.__version__}, GPU: " + ", ".join(torch.cuda.get_device_name(k) for k in range(torch.cuda.device_count()))
              if gpu else "torch without a GPU")
    except ImportError:
        gpu = False
        print("torch is missing (run with: uv run python run_v2.py)")
    if not gpu and not args.allow_cpu:
        sys.exit("no GPU visible: stopping (--allow-cpu to run anyway)")
    need = [HERE / INPUT / f for f in ("train.npz", "test_users.json", "train_users.json", "items.json",
                                       "item_reviews.00.jsonl.gz")]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        sys.exit("missing (git pull?):\n  " + "\n  ".join(missing))

    if "imdb" in args.stages:
        say("stage imdb: EXP3RT's IMDB test, the persona inside the backbone, our reranker")
        out = HERE / "runs" / "imdb"
        if (out / "eval_imdb.md").exists():
            print(f"skipped: {out / 'eval_imdb.md'} exists")
        else:
            log = out / "imdb_log.txt"
            if tee(py("imdb.py", "fetch"), log) or tee(py("imdb.py", "run", "--out", out, "--encoder", ENCODER,
                                                          "--seeds", args.seeds), log):
                sys.exit(f"imdb stage failed; see {log}")
            push(["runs/imdb"], "IMDB: EXP3RT's top-k test, persona inside LightGCN / MF (ablation), our reranker", args)

    if "exp3rt" in args.stages:
        say("stage exp3rt: EXP3RT's student fine-tuned (exact and clean), then the top-20 lists scored")
        ok = exp3rt_stage(args, torch.cuda.device_count() if gpu else 1)
        push(["runs/imdb"], "IMDB: EXP3RT reproduced (Llama 3.1 8B student, exact and clean training data)"
             + ("" if ok else " (FAILED, logs only)"), args)
        if not ok:
            print("exp3rt stage FAILED: see runs/imdb/exp3rt_*/train.txt or score.txt; going on with the next stages")

    if "smoke" in args.stages:
        say("stage smoke: run.py --smoke on the larger data (Llama 3.1 8B through Ollama)")
        tmp = HERE / "runs" / "v2_smoke"
        code = tee(py("run.py", "--smoke", "--input", INPUT, "--out", tmp), HERE / "runs" / "v2_smoke.txt")
        if code:
            sys.exit(f"smoke stage failed (exit {code}); the output is in runs/v2_smoke.txt and {tmp}. Nothing pushed.")
        shutil.rmtree(tmp, ignore_errors=True)
        (HERE / "runs" / "v2_smoke.txt").unlink(missing_ok=True)
        print("smoke ok")

    if "lgcn" in args.stages:
        say("stage lgcn: LightGCN + Llama 3.1 8B personas on the larger data")
        run = HERE / RUNS["lgcn"]
        if (run / "eval.md").exists():
            print(f"skipped: {run / 'eval.md'} exists")
        else:
            code = tee(py("run.py", "--input", INPUT, "--out", run, "--lgcn-minutes", "120", "--item-profiles", "skip"),
                       run / "run_v2.txt")
            push([RUNS["lgcn"]], "v2 LightGCN + Llama 3.1 8B personas" + (" (FAILED)" if code else ""), args)
            if code:
                sys.exit(f"lgcn stage failed (exit {code})")

    if "bases" in args.stages:
        for model in ("pop", "sasrec"):
            run = HERE / RUNS[model]
            say(f"stage bases: {model}")
            if (run / "candidates.jsonl").exists():
                print(f"skipped: {run / 'candidates.jsonl'} exists")
            else:
                extra = ["--max-minutes", "120"] if model == "sasrec" else []
                if tee(py("base_models.py", "--model", model, "--input", INPUT, "--out", run, *extra), run / "base_log.txt"):
                    sys.exit(f"{model} failed")
            for f in ("personas.jsonl", "personas_train.jsonl"):
                src = HERE / RUNS["lgcn"] / f
                if not src.exists():
                    sys.exit(f"{src} is missing: run the lgcn stage first")
                shutil.copy(src, run / f)
        push([RUNS["pop"], RUNS["sasrec"]], "v2 popularity and SASRec lists", args)

    if "eval" in args.stages:
        for name, run in RUNS.items():
            say(f"stage eval: {run} with {ENCODER}")
            if (HERE / run / "eval_qwen8b.md").exists():
                print("skipped: eval_qwen8b.md exists")
                continue
            code = tee(py("evaluate.py", run, "--input", INPUT, "--encoder", ENCODER, "--alpha", "0.1", "--beta", "0.1",
                          "--tag", "qwen8b", "--no-lgbm", "--show", "0"), HERE / run / "evaluate_qwen8b.txt")
            push([run], f"v2 {name}: Qwen3-Embedding-8B evaluation" + (" (FAILED)" if code else ""), args)
    say(f"done in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
