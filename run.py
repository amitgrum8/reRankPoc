"""One command for the whole PoC on a machine with an NVIDIA GPU (with --mock, on any machine).

  python run.py --smoke          # 1 user, LightGCN 2 min: proves the setup end to end (about 10-15 min)
  python run.py                  # every test user, LightGCN up to 30 min
  python run.py --users 5        # the first 5 test users
  python run.py --mock           # no GPU, no LLM: fake answers; checks the plumbing only
  python run.py --out runs/x     # run again into the same folder: finished stages are skipped

Stages: environment check -> LightGCN (top 500 per test user) -> Ollama server + model -> reasoning check
and timed calls -> EXP3RT personas for the test users, then for the reranker's training users (train_users.json;
--train-users 0 skips them) -> item profiles (what reviewers praised and criticized, for the items in this run's
lists; needs data/rr_input/item_reviews.jsonl.gz from prep_items.py; shared by all runs in data/item_profiles/;
--item-profiles all|lists|skip, --smoke skips) -> evaluation (formulas R1-R5, R8-R9 with the item profiles,
LightGBM R7 trained on the training users, and the swapped-persona control).
Everything lands in runs/<name>/. Every LLM answer is cached in runs/cache/, shared by all runs, so repeating
a run costs no LLM time.

Ollama is started by this script on its own port (11435) with the settings the prompts need (16k context,
4 parallel requests), so the Ollama tray app, if any, does not matter.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
T_START = time.time()


class Tee:
    """stdout that also appends to a file."""
    def __init__(self, path):
        self.console, self.file = sys.__stdout__, open(path, "a", encoding="utf-8")

    def write(self, text):
        try:
            self.console.write(text)
        except UnicodeEncodeError:
            enc = self.console.encoding or "ascii"
            self.console.write(text.encode(enc, "replace").decode(enc))
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


def say(msg):
    print(f"\n=== {msg}  [{(time.time() - T_START) / 60:.1f} min]", flush=True)


def run(cmd, env=None, must=True):
    """Runs a command, streaming its output; stops the script if it fails (must), or just says so."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    env = dict(env or os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env, cwd=str(HERE))
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if p.returncode != 0:
        if must:
            sys.exit(f"'{cmd[0]}' failed with exit code {p.returncode}")
        print(f"'{cmd[-1] if len(cmd) < 3 else cmd[2]}' failed with exit code {p.returncode}; going on", flush=True)
    return p.returncode


# ------------------------------------------------------------------ environment
def check_env(args):
    say("environment")
    print(f"python {sys.version.split()[0]} | {platform.platform()}")
    try:
        import torch
    except ImportError:
        sys.exit("torch is missing:\n  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128")
    import numpy, openai  # noqa: F401  (fail early if missing)
    print(f"torch {torch.__version__} | numpy {numpy.__version__} | openai {openai.__version__}")
    if torch.cuda.is_available():
        for k in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(k)
            print(f"gpu {k}: {p.name}, {p.total_memory / 2**30:.0f} GB")
    elif args.mock or args.cpu:
        print("no CUDA GPU: LightGCN runs on the CPU (slow; fine for --mock)")
    else:
        sys.exit("torch sees no CUDA GPU. Install the CUDA build:\n"
                 "  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128\n"
                 "or pass --cpu to train LightGCN on the CPU (about 17 min per epoch).")
    for name in ("train.npz", "test_users.json", "items.json"):
        if not (args.input / name).exists():
            sys.exit(f"{args.input / name} is missing (run prep_data.py, or git pull the data)")


# ------------------------------------------------------------------ Ollama
def find_ollama():
    exe = shutil.which("ollama")
    if not exe and os.name == "nt":
        cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        exe = str(cand) if cand.exists() else None
    if not exe:
        sys.exit("ollama is not installed. Get it from https://ollama.com/download (no admin rights needed), "
                 "then open a new terminal and run this again.")
    return exe


def http_get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


class Ollama:
    def __init__(self, exe, port, log_path):
        self.exe, self.host = exe, f"127.0.0.1:{port}"
        self.url = f"http://{self.host}"
        self.env = dict(os.environ, OLLAMA_HOST=self.host, OLLAMA_NUM_PARALLEL="4", OLLAMA_CONTEXT_LENGTH="16384",
                        OLLAMA_KEEP_ALIVE="-1")
        self.proc = None
        try:
            print(f"ollama already answering on {self.host}: {http_get(self.url + '/api/version')}")
            return
        except Exception:
            pass
        self.log = open(log_path, "a")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen([exe, "serve"], env=self.env, stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=flags)
        for _ in range(120):
            try:
                print(f"ollama {http_get(self.url + '/api/version')} on {self.host}")
                return
            except Exception:
                time.sleep(1)
        sys.exit(f"ollama serve did not start; see {log_path}")

    def has_model(self, model):
        tags = json.loads(http_get(self.url + "/api/tags"))
        names = {m["name"] for m in tags.get("models", [])}
        return model in names or f"{model}:latest" in names

    def pull(self, model):
        """Downloads the model through the server's API, printing progress every 15 s (a stopped download
        resumes where it was)."""
        say(f"pulling {model} (once; about 13 GB)")
        req = urllib.request.Request(self.url + "/api/pull", data=json.dumps({"name": model, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        t0, last = time.time(), 0.0
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                d = json.loads(raw)
                if "error" in d:
                    sys.exit(f"pull failed: {d['error']}")
                if d.get("total") and time.time() - last >= 15:
                    done, total = d.get("completed", 0), d["total"]
                    print(f"  {d.get('status', '')}: {done / 2**30:.2f} / {total / 2**30:.2f} GB "
                          f"({100 * done / total:.0f}%)  {(time.time() - t0) / 60:.1f} min", flush=True)
                    last = time.time()
        print(f"pulled in {(time.time() - t0) / 60:.1f} min", flush=True)
        if not self.has_model(model):
            sys.exit("the pull finished but the model is not listed; see the ollama log")

    def stop(self):
        if self.proc is None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            self.proc.terminate()


# ------------------------------------------------------------------ LLM check
def llm_check(args, base_url, out, n_users):
    """The same prompt at reasoning 'low' and 'high' (high must use clearly more output tokens), then 8 timed
    calls at 'high' with 4 in parallel. Returns the extra personas.py flags (JSON mode off if it fails)."""
    sys.path.insert(0, str(HERE))
    import llm as L, prompts as P, personas as PS
    say("reasoning check")
    users = json.loads((args.input / "test_users.json").read_text(encoding="utf-8"))["users"]
    hist = users[0]["history"]
    msgs = [P.extract(h.get("review_title", ""), h.get("text", "")) for h in hist[:9]]
    for effort in ("low", "high"):
        c = L.LLM(base_url, args.model, out / "cache_check", effort)
        t0 = time.time()
        c.call(msgs[0], {"role": "extract"})
        line = c.log_lines[-1]
        print(f"reasoning {effort:4s}: {line['completion_tokens']} output tokens, {time.time() - t0:.0f}s "
              f"(cached: {line['cached']})")

    def timed(json_mode):
        c = L.LLM(base_url, args.model, args.cache, args.reasoning, json_mode)
        t0 = time.time()
        with ThreadPoolExecutor(4) as ex:
            outs = list(ex.map(lambda m: c.call(m, {"role": "extract"}), msgs[1:9]))
        ok = sum(PS.check(L.parse_json(o)) is not None for o in outs)
        return time.time() - t0, ok, c

    say("timed calls: 8 extractions, 4 in parallel")
    wall, ok, c = timed(not args.no_json_mode)
    flags = ["--no-json-mode"] if args.no_json_mode else []
    if ok < 6 and not args.no_json_mode:
        print(f"JSON mode: only {ok}/8 valid -> trying without it")
        wall, ok, c = timed(False)
        flags = ["--no-json-mode"]
    s = c.stats
    n_calls = sum(len(u["history"]) + 1 for u in users[:n_users])
    if s["new"]:
        per_call = wall / 8
        print(f"{ok}/8 valid | 8 calls in {wall:.0f}s ({per_call:.1f}s per call effective) | "
              f"{s['completion_tokens'] / 8:.0f} output tokens per call")
        print(f"projected for {n_calls} calls ({n_users} users): {n_calls * per_call / 3600:.1f} h")
    else:
        print(f"{ok}/8 valid | all cached, no timing")
    return flags


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="run folder (default runs/<date_time>)")
    ap.add_argument("--input", default=str(HERE / "data" / "rr_input"))
    ap.add_argument("--cache", default=str(HERE / "runs" / "cache"), help="LLM cache shared by all runs")
    ap.add_argument("--users", default="all", help="'all' or the first n test users")
    ap.add_argument("--train-users", default="all",
                    help="'all', the first n, or 0: the reranker's training users (train_users.json) to run personas for")
    ap.add_argument("--lgcn-minutes", type=float, default=None, help="LightGCN time limit (default 30; --smoke 2)")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--hours", type=float, default=0, help="stop new LLM calls after this many hours (0 = never)")
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--item-profiles", default=None, choices=("lists", "all", "skip"),
                    help="LLM item profiles for the items in this run's lists (default), every item, or none (--smoke default)")
    ap.add_argument("--smoke", action="store_true", help="1 test user, 3 training users, LightGCN 2 min")
    ap.add_argument("--mock", action="store_true", help="fake LLM answers, no Ollama, CPU allowed")
    ap.add_argument("--cpu", action="store_true", help="allow LightGCN on the CPU")
    args = ap.parse_args()
    if args.smoke:
        args.users, args.train_users = "1", "3"
        args.lgcn_minutes = args.lgcn_minutes or 2
    args.lgcn_minutes = args.lgcn_minutes or 30
    args.item_profiles = args.item_profiles or ("skip" if args.smoke else "lists")
    args.input, args.cache = Path(args.input), Path(args.cache)
    out = Path(args.out) if args.out else HERE / "runs" / time.strftime("%Y-%m-%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "run.log")       # everything printed also lands in the run folder
    print(f"run folder {out} | users {args.users} | LightGCN up to {args.lgcn_minutes:g} min | model {args.model} "
          f"| reasoning {args.reasoning}" + (" | MOCK" if args.mock else ""))
    check_env(args)

    # 1. LightGCN
    if (out / "candidates.jsonl").exists():
        say("LightGCN: skipped, candidates.jsonl exists")
    else:
        say("LightGCN")
        run([sys.executable, "-u", HERE / "lightgcn.py", "--input", args.input, "--out", out,
             "--max-minutes", args.lgcn_minutes])

    # 2-4. personas: the test users, then the reranker's training users (train_users.json), on the same server
    def count(name, want):
        n = len(json.loads((args.input / name).read_text(encoding="utf-8"))["users"])
        return n if want == "all" else min(n, int(want))

    n_users = count("test_users.json", args.users)
    n_train = count("train_users.json", args.train_users) if (args.input / "train_users.json").exists() else 0
    stages = [("test", args.users, n_users, "personas.jsonl")]
    if n_train:
        stages.append(("train", args.train_users, n_train, "personas_train.jsonl"))
    todo = [s for s in stages if not (out / s[3]).exists()]
    for s in stages:
        if s not in todo:
            say(f"personas ({s[0]}): skipped, {s[3]} exists")
    reviews = args.input / "item_reviews.jsonl.gz"
    want_profiles = args.item_profiles != "skip" and reviews.exists()
    if args.item_profiles != "skip" and not reviews.exists():
        say(f"item profiles: skipped, {reviews} is missing (run prep_items.py and push it)")
    if todo or want_profiles:
        server, flags, base_url = None, [], "http://127.0.0.1:1/v1"
        if not args.mock:
            server = Ollama(find_ollama(), args.port, out / "ollama.log")
            base_url = server.url + "/v1"
            if not server.has_model(args.model):
                server.pull(args.model)
            flags = llm_check(args, base_url, out, n_users)
        t_llm = time.time()
        for set_, want, n, _ in todo:
            say(f"personas for {n} {set_} user(s)")
            cmd = [sys.executable, "-u", HERE / "personas.py", "--input", args.input, "--out", out, "--cache", args.cache,
                   "--base-url", base_url, "--model", args.model, "--reasoning-effort", args.reasoning,
                   "--set", set_, "--users", want, "--threads", args.threads] + flags
            if args.hours:                           # one time budget for both sets; the test users come first
                left = max(args.hours - (time.time() - t_llm) / 3600, 0.1)
                cmd += ["--deadline-hours", left, "--start-deadline-hours", max(left - 0.5, 0.1)]
            if args.mock:
                cmd += ["--mock"]
            run(cmd, must=set_ == "test")             # no training personas: the evaluation still runs, without R7
        if want_profiles:                                 # 4b. item profiles, for the items in this run's lists
            say(f"item profiles ({'every item' if args.item_profiles == 'all' else 'the items in this run'})")
            cmd = [sys.executable, "-u", HERE / "item_profiles.py", "--input", args.input,
                   "--out", HERE / "data" / "item_profiles", "--cache", args.cache, "--base-url", base_url,
                   "--model", args.model, "--reasoning-effort", args.reasoning, "--threads", args.threads,
                   "--items", args.item_profiles, "--run", out] + flags
            if args.hours:
                cmd += ["--deadline-hours", max(args.hours - (time.time() - t_llm) / 3600, 0.1)]
            if args.mock:
                cmd += ["--mock"]
            run(cmd, must=False)                          # without profiles the evaluation runs without R8-R9
        if server:
            server.stop()

    # 5. evaluation
    say("evaluation")
    run([sys.executable, "-u", HERE / "evaluate.py", out, "--input", args.input])
    say("done")
    for f in sorted(out.glob("*")):
        if f.is_file():
            print(f"{f.name:20s} {f.stat().st_size:>12,} bytes")
    print(f"\nresults in {out}  (eval.md, personas.jsonl, candidates.jsonl, summary.json, lightgcn.json)")


if __name__ == "__main__":
    main()
