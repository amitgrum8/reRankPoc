"""Step 4 (local): metrics of the LightGCN lists, with the slot where a reranker plugs in.

  python evaluate.py run                       # run/ holds candidates.jsonl (+ personas.jsonl)
  python evaluate.py run --input data/rr_input --show 2

For every test user, the list = LightGCN's top 100 + the user's targets (their first reviews after the
cutoff) put into it. A reranker gives every item in that list a score; each target's rank is its place
among the 100 candidates (other targets don't count). Targets are split by stars:
  liked (4-5)     HR@10, NDCG@10, mean rank        (higher HR/NDCG and lower rank = better)
  disliked (1-2)  avoid@10 (not in the top 10), mean rank   (higher = better)
  3 stars         mean rank, for reference
  pairs           within-user AUC: share of (liked, disliked) target pairs where the liked one scores
                  higher (ties 0.5); mean over users. 0.5 = the ranker can't tell them apart.
  retrieved       share of targets LightGCN put in its top 100 at all (the same for every reranker)

A reranker is a function rerank(user) -> {item: score} over user["list"] (see RERANKERS at the bottom).
"lightgcn" keeps LightGCN's order; the persona formulas R1-R5 come from rerank.py (similarity features between
the persona lines and the item text, written to <run>/features.jsonl) when sentence-transformers is installed.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_users(run, inp):
    cands = load_jsonl(run / "candidates.jsonl")
    personas = {p["user_id"]: p for p in load_jsonl(run / "personas.jsonl")} if (run / "personas.jsonl").exists() else {}
    test = {tu["user_id"]: tu for tu in json.loads((inp / "test_users.json").read_text(encoding="utf-8"))["users"]}
    users = []
    for c in cands:
        if personas and c["user_id"] not in personas:
            continue                                  # evaluate only users who got a persona
        base = {i: s for i, s in c["candidates"]}
        cand_ids = [i for i, _ in c["candidates"]]
        for t in c["targets"]:
            base.setdefault(t["i"], t["score"])       # a target outside the top 100 enters with its own score
        users.append({"user_id": c["user_id"], "u": c["u"], "candidates": cand_ids, "base": base,
                      "list": list(base), "targets": c["targets"], "persona": personas.get(c["user_id"]),
                      "history": test.get(c["user_id"], {}).get("history", [])})
    return users


def evaluate(users, rerank, k=10):
    liked_hr, liked_ndcg, liked_rank, dis_avoid, dis_rank, three_rank, aucs = [], [], [], [], [], [], []
    retrieved = []
    for user in users:
        s = rerank(user)
        cand = set(user["candidates"])
        cand_scores = np.array([s[i] for i in user["candidates"]])
        tscore = {}
        for t in user["targets"]:
            st = s[t["i"]]
            others = cand_scores[[i != t["i"] for i in user["candidates"]]] if t["i"] in cand else cand_scores
            rank = int((others > st).sum()) + 1
            tscore[t["i"]] = st
            retrieved.append(t["i"] in cand)
            if t["rating"] >= 4:
                liked_hr.append(rank <= k); liked_ndcg.append(1 / math.log2(rank + 1) if rank <= k else 0.0)
                liked_rank.append(rank)
            elif t["rating"] <= 2:
                dis_avoid.append(rank > k); dis_rank.append(rank)
            else:
                three_rank.append(rank)
        liked = [tscore[t["i"]] for t in user["targets"] if t["rating"] >= 4]
        disliked = [tscore[t["i"]] for t in user["targets"] if t["rating"] <= 2]
        if liked and disliked:
            pairs = [1.0 if a > b else 0.5 if a == b else 0.0 for a in liked for b in disliked]
            aucs.append(sum(pairs) / len(pairs))
    m = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {"users": len(users), "liked_n": len(liked_hr), f"liked_HR@{k}": m(liked_hr), f"liked_NDCG@{k}": m(liked_ndcg),
            "liked_mean_rank": m(liked_rank), "disliked_n": len(dis_rank), f"disliked_avoid@{k}": m(dis_avoid),
            "disliked_mean_rank": m(dis_rank), "three_n": len(three_rank), "three_mean_rank": m(three_rank),
            "pair_AUC": m(aucs), "pair_users": len(aucs), "retrieved_in_top100": m(retrieved)}


def show_user(user, items, rerank, n=10):
    title = lambda i: (items[i]["title"] or items[i]["asin"])[:70]
    print(f"\n=== user {user['user_id']}")
    if user["persona"]:
        print("  likes:    " + " | ".join(user["persona"]["like"]))
        print("  dislikes: " + " | ".join(user["persona"]["dislike"]))
    s = rerank(user)
    feat = user.get("feat")
    sims = (lambda i: f"  like {feat[i]['like_max']:.2f} dis {feat[i]['dis_max']:.2f}  ") if feat else (lambda i: "  ")
    if feat:
        print(f"  tau (dislike threshold) {user['tau']:.2f}; like/dis = closest like/dislike line (cosine)")
    order = sorted(user["candidates"], key=lambda i: -s[i])
    print(f"  top {n} of the list:")
    for r, i in enumerate(order[:n], 1):
        print(f"    {r:2d}.{sims(i)}{title(i)}")
    print("  targets (stars, rank among the 100):")
    cand_scores = {i: s[i] for i in user["candidates"]}
    for t in user["targets"]:
        rank = sum(v > s[t["i"]] for i, v in cand_scores.items() if i != t["i"]) + 1
        print(f"    {t['rating']}* rank {rank:3d}{sims(t['i'])}{title(t['i'])}")


# ------------------------------------------------------------------------------------------------- rerankers
def rerank_lightgcn(user):
    """No reranking: LightGCN's own scores."""
    return user["base"]


RERANKERS = {"lightgcn": rerank_lightgcn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--input", default=str(HERE / "data" / "rr_input"))
    ap.add_argument("--show", type=int, default=1, help="print this many example users")
    ap.add_argument("--show-with", default="R3_soft", help="the reranker whose list the examples show")
    ap.add_argument("--no-rerank", action="store_true", help="LightGCN column only (no encoder needed)")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the like similarity (rerank.py)")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the dislike penalty (rerank.py)")
    ap.add_argument("--gamma", type=float, default=1.0, help="dislike weight inside the Rocchio vector (rerank.py)")
    args = ap.parse_args()
    run, inp = Path(args.run), Path(args.input)
    users = build_users(run, inp)
    items = json.loads((inp / "items.json").read_text(encoding="utf-8"))
    print(f"{len(users)} users" + (" (with a persona)" if users and users[0]["persona"] else ""))

    rerankers = dict(RERANKERS)
    if users and users[0]["persona"] and not args.no_rerank:
        try:
            import rerank as R
        except ImportError as e:                      # no sentence-transformers here: the LightGCN column only
            print(f"persona reranker skipped ({e}); pip install sentence-transformers")
        else:
            R.PARAMS.update(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            R.attach(users, items, R.Encoder())
            R.write_features(users, run / "features.jsonl")
            rerankers.update(R.RERANKERS)
            print(f"features of {sum(len(u['list']) for u in users)} (user, item) rows -> features.jsonl; "
                  f"alpha {args.alpha} beta {args.beta} gamma {args.gamma}")

    rows = {name: evaluate(users, f) for name, f in rerankers.items()}
    keys = list(next(iter(rows.values())).keys())
    lines = ["| metric | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    for key in keys:
        vals = [rows[r][key] for r in rows]
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        lines.append(f"| {key} | " + " | ".join(fmt(v) for v in vals) + " |")
    table = "\n".join(lines)
    print(table)
    (run / "eval.md").write_text(table + "\n", encoding="utf-8")
    for user in users[:args.show]:
        show_user(user, items, rerankers.get(args.show_with, rerank_lightgcn))


if __name__ == "__main__":
    main()
