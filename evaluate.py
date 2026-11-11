"""Step 4 (local, CPU): metrics of every reranker on the test users -> <run>/eval.md.

  python evaluate.py run                       # run/ holds candidates.jsonl (+ personas.jsonl)
  python evaluate.py run --input data/rr_input --show 2 --show-with R2_like-dis

For every test user, the list = LightGCN's top k (500) + the user's targets (their first reviews after the
cutoff) put into it. A reranker gives every item in that list a score; each target's rank is its place
among the k candidates (other targets don't count). Targets are split by stars:
  liked (4-5)     HR@10, NDCG@10, mean rank; rank_change = rank - rank under LightGCN (negative = moved up),
                  moved_up = share of liked targets the reranker moved up
  disliked (1-2)  avoid@10 (not in the top 10), mean rank; rank_change (positive = moved down), moved_down
  separation      mean rank of the disliked targets - mean rank of the liked ones, per user with both
  3 stars         mean rank, for reference
  pair_AUC        within-user: share of (liked, disliked) target pairs where the liked one scores higher (ties
                  0.5), mean over users. 0.5 = the ranker can't tell them apart. The thesis metric (BLUEPRINT V3).
  retrieved       share of targets LightGCN put in its list at all (the same for every reranker)
A second table gives every reranker's pair_AUC minus LightGCN's, with a 95% interval from a paired bootstrap
over users. HR@10 stays near zero for every reranker because most targets enter below all k candidates; the
rank changes and pair_AUC are the numbers that show whether the persona did anything.

A reranker is a function rerank(user) -> {item: score} over user["list"] (see RERANKERS at the bottom).
"lightgcn" keeps LightGCN's order; the formulas R1-R5 come from rerank.py (similarity features between the
persona lines and the item text) when sentence-transformers is installed; R7_lgbm from train.py when the run
holds personas for the training users (personas_train.jsonl).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def target_ranks(user, s):
    """Every target's rank among the user's candidates under the scores s: {item: rank}, 1 = top."""
    cand = user["candidates"]
    cand_scores = np.array([s[i] for i in cand])
    out = {}
    for t in user["targets"]:
        st = s[t["i"]]
        others = cand_scores[[i != t["i"] for i in cand]] if t["i"] in user["cand_set"] else cand_scores
        out[t["i"]] = int((others > st).sum()) + 1
    return out


def build_users(run, inp, set_="test"):
    """The users of a set (test or train) that have a persona, with their list, base scores and targets."""
    sfx = "" if set_ == "test" else "_train"
    cands = load_jsonl(run / f"candidates{sfx}.jsonl")
    pfile = run / f"personas{sfx}.jsonl"
    personas = {p["user_id"]: p for p in load_jsonl(pfile)} if pfile.exists() else {}
    meta = {tu["user_id"]: tu for tu in json.loads((inp / f"{set_}_users.json").read_text(encoding="utf-8"))["users"]}
    users = []
    for c in cands:
        if personas and c["user_id"] not in personas:
            continue                                  # evaluate only users who got a persona
        base = {i: s for i, s in c["candidates"]}
        cand_ids = [i for i, _ in c["candidates"]]
        for t in c["targets"]:
            base.setdefault(t["i"], t["score"])       # a target outside the top k enters with its own score
        user = {"user_id": c["user_id"], "u": c["u"], "candidates": cand_ids, "cand_set": set(cand_ids), "base": base,
                "list": list(base), "targets": c["targets"], "persona": personas.get(c["user_id"]),
                "history": meta.get(c["user_id"], {}).get("history", [])}
        user["base_ranks"] = target_ranks(user, base)
        users.append(user)
    return users


def evaluate(users, rerank, k=10):
    liked_hr, liked_ndcg, liked_rank, liked_chg = [], [], [], []
    dis_avoid, dis_rank, dis_chg, three_rank, aucs, sep, retrieved = [], [], [], [], [], [], []
    for user in users:
        s = rerank(user)
        ranks = target_ranks(user, s)
        for t in user["targets"]:
            rank, change = ranks[t["i"]], ranks[t["i"]] - user["base_ranks"][t["i"]]
            retrieved.append(t["i"] in user["cand_set"])
            if t["rating"] >= 4:
                liked_hr.append(rank <= k); liked_ndcg.append(1 / math.log2(rank + 1) if rank <= k else 0.0)
                liked_rank.append(rank); liked_chg.append(change)
            elif t["rating"] <= 2:
                dis_avoid.append(rank > k); dis_rank.append(rank); dis_chg.append(change)
            else:
                three_rank.append(rank)
        liked = [t for t in user["targets"] if t["rating"] >= 4]
        disliked = [t for t in user["targets"] if t["rating"] <= 2]
        if liked and disliked:
            pairs = [1.0 if s[a["i"]] > s[b["i"]] else 0.5 if s[a["i"]] == s[b["i"]] else 0.0 for a in liked for b in disliked]
            aucs.append(sum(pairs) / len(pairs))
            sep.append(np.mean([ranks[t["i"]] for t in disliked]) - np.mean([ranks[t["i"]] for t in liked]))
    m = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {"users": len(users), "list_size": len(users[0]["candidates"]) if users else 0,
            "liked_n": len(liked_hr), f"liked_HR@{k}": m(liked_hr), f"liked_NDCG@{k}": m(liked_ndcg),
            "liked_mean_rank": m(liked_rank), "liked_rank_change": m(liked_chg), "liked_moved_up": m([x < 0 for x in liked_chg]),
            "disliked_n": len(dis_rank), f"disliked_avoid@{k}": m(dis_avoid), "disliked_mean_rank": m(dis_rank),
            "disliked_rank_change": m(dis_chg), "disliked_moved_down": m([x > 0 for x in dis_chg]),
            "separation": m(sep), "three_n": len(three_rank), "three_mean_rank": m(three_rank),
            "pair_AUC": m(aucs), "pair_users": len(aucs), "retrieved_in_list": m(retrieved)}


def pair_aucs(users, rerank):
    """Per-user pair AUC, aligned with users (nan for a user without both a liked and a disliked target)."""
    out = []
    for user in users:
        s = rerank(user)
        liked = [s[t["i"]] for t in user["targets"] if t["rating"] >= 4]
        disliked = [s[t["i"]] for t in user["targets"] if t["rating"] <= 2]
        out.append(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in liked for b in disliked])
                   if liked and disliked else np.nan)
    return np.array(out)


def bootstrap_diff(a, b, n=2000, seed=42):
    """Mean of a - b over users, with a 95% interval from a paired bootstrap over users."""
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


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
    print(f"  targets (stars, rank among the {len(user['candidates'])}, rank under LightGCN):")
    ranks = target_ranks(user, s)
    for t in user["targets"]:
        print(f"    {t['rating']}* rank {ranks[t['i']]:3d} (lightgcn {user['base_ranks'][t['i']]:3d}){sims(t['i'])}{title(t['i'])}")


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
    ap.add_argument("--show-with", default="R2_like-dis", help="the reranker whose list the examples show")
    ap.add_argument("--no-rerank", action="store_true", help="LightGCN column only (no encoder needed)")
    ap.add_argument("--no-train", action="store_true", help="skip the LightGBM reranker even if personas_train.jsonl exists")
    ap.add_argument("--write-features", action="store_true", help="write the test users' feature rows to features.jsonl")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the like similarity / model score (rerank.py)")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the dislike penalty (rerank.py)")
    ap.add_argument("--gamma", type=float, default=1.0, help="dislike weight inside the Rocchio vector (rerank.py)")
    args = ap.parse_args()
    run, inp = Path(args.run), Path(args.input)
    users = build_users(run, inp)
    items = json.loads((inp / "items.json").read_text(encoding="utf-8"))
    print(f"{len(users)} test users" + (" (with a persona)" if users and users[0]["persona"] else "")
          + (f", lists of {len(users[0]['candidates'])}" if users else ""))

    rerankers = dict(RERANKERS)
    if users and users[0]["persona"] and not args.no_rerank:
        try:
            import rerank as R
        except ImportError as e:                      # no sentence-transformers here: the LightGCN column only
            print(f"persona reranker skipped ({e}); pip install sentence-transformers")
        else:
            R.PARAMS.update(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            enc = R.Encoder()
            R.attach(users, items, enc)
            if args.write_features:
                R.write_features(users, run / "features.jsonl")
            rerankers.update(R.RERANKERS)
            print(f"features of {sum(len(u['list']) for u in users)} (user, item) rows; "
                  f"alpha {args.alpha} beta {args.beta} gamma {args.gamma}")
            if (run / "personas_train.jsonl").exists() and not args.no_train:
                import train as T
                if not (run / "candidates_train.jsonl").exists():
                    n = T.write_candidates(run, inp, k=len(users[0]["candidates"]))
                    print(f"candidates_train.jsonl: lists for {n} training users from embeddings.npz")
                users_train = build_users(run, inp, "train")
                R.attach(users_train, items, enc)
                model, info = T.fit(users_train, run)
                rerankers["R7_lgbm"] = T.reranker(model, alpha=args.alpha)
                print(f"LightGBM on {info['train_users']} training users ({info['rows']} rows, {info['liked_rows']} liked, "
                      f"{info['disliked_rows']} disliked targets): {info['trees']} trees, validation NDCG@10 "
                      f"{info['val_ndcg@10']} on {info['val_users']} users -> lgbm.txt, lgbm.json")
                top = sorted(info["importance_gain"].items(), key=lambda kv: -kv[1])[:5]
                print("  most used features (gain): " + ", ".join(f"{k} {v:.0f}" for k, v in top))

    rows = {name: evaluate(users, f) for name, f in rerankers.items()}
    keys = list(next(iter(rows.values())).keys())
    lines = ["| metric | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(fmt(rows[r][key]) for r in rows) + " |")
    if len(rerankers) > 1:
        base_auc = pair_aucs(users, rerank_lightgcn)
        lines += ["", "pair_AUC minus LightGCN's, paired bootstrap over users (95% interval):", "",
                  "| reranker | mean diff | 2.5% | 97.5% |", "|---|---|---|---|"]
        for name, f in rerankers.items():
            if name != "lightgcn":
                d, lo, hi = bootstrap_diff(pair_aucs(users, f), base_auc)
                lines.append(f"| {name} | {d:+.3f} | {lo:+.3f} | {hi:+.3f} |")
    table = "\n".join(lines)
    print(table)
    (run / "eval.md").write_text(table + "\n", encoding="utf-8")
    for user in users[:args.show]:
        show_user(user, items, rerankers.get(args.show_with, rerank_lightgcn))


if __name__ == "__main__":
    main()
