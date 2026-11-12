"""Step 4 (local, CPU): metrics of every reranker on the test users -> <run>/eval.md.

  python evaluate.py run                       # run/ holds candidates.jsonl (+ personas.jsonl)
  python evaluate.py run --input data/rr_input --show 2 --show-with R1_like

For every test user, the list = LightGCN's top k (500) + the user's targets (their first reviews after the
cutoff) put into it. A reranker gives every item in that list a score; each target's rank is its place
among the k candidates (other targets don't count).

The question is only how much the reranker moved the user's interacted items (targets, any rating) inside
the list of k candidates:
  in_list_*       targets that LightGCN put in its top k: their mean rank under LightGCN and under the reranker,
                  mean rank change (negative = moved up), and how many moved up / stayed / moved down
  in_list_top10   how many of them end in the top 10
  out_of_list     targets below LightGCN's top k (they enter at rank k+1 and can only move up);
                  out_pulled_in = how many the reranker lifted above at least one candidate
  all_mean_rank   mean rank of every target
A second table gives every reranker's mean in-list rank change per user, with a 95% interval from a bootstrap
over users (0 = no change). A third table is the control: the same rerankers with every user given another
user's persona (a fixed derangement). A persona that is not the user's own should not move their items; what
it still moves is the lift any text gives, not personalization.

A reranker is a function rerank(user) -> {item: score} over user["list"] (see RERANKERS at the bottom).
"lightgcn" keeps LightGCN's order; the formulas R1-R5 come from rerank.py (similarity features between the
persona lines and the item text) when sentence-transformers is installed, R8-R9 when item profiles exist
(data/item_profiles/item_profiles.jsonl from item_profiles.py); R7_lgbm from train.py when the run holds
personas for the training users (personas_train.jsonl).
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


def swap_personas(users, seed=42):
    """Copies of the users where each one carries another user's persona (a fixed derangement)."""
    order = np.random.default_rng(seed).permutation(len(users))
    donor = {users[order[k]]["user_id"]: users[order[(k + 1) % len(order)]]["persona"] for k in range(len(order))}
    return [{**u, "persona": donor[u["user_id"]]} for u in users]


def evaluate(users, rerank, k=10):
    """Movement of the user's interacted items inside the candidate list, against LightGCN's order."""
    in_base, in_new, chg, all_new, out_n, pulled = [], [], [], [], 0, 0
    for user in users:
        ranks = target_ranks(user, rerank(user))
        n = len(user["candidates"])
        for t in user["targets"]:
            r0, r1 = user["base_ranks"][t["i"]], ranks[t["i"]]
            all_new.append(r1)
            if t["i"] in user["cand_set"]:
                in_base.append(r0); in_new.append(r1); chg.append(r1 - r0)
            else:
                out_n += 1; pulled += r1 <= n
    m = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {"users": len(users), "list_size": len(users[0]["candidates"]) if users else 0,
            "targets": len(all_new), "in_list": len(in_base),
            "in_list_mean_rank_lightgcn": m(in_base), "in_list_mean_rank": m(in_new), "in_list_rank_change": m(chg),
            "in_list_moved_up": sum(c < 0 for c in chg), "in_list_stayed": sum(c == 0 for c in chg),
            "in_list_moved_down": sum(c > 0 for c in chg), f"in_list_top{k}": sum(r <= k for r in in_new),
            "out_of_list": out_n, "out_pulled_in": pulled, "all_mean_rank": m(all_new)}


def user_changes(users, rerank):
    """Per-user mean rank change of the in-list targets, aligned with users (nan without in-list targets)."""
    out = []
    for user in users:
        ranks = target_ranks(user, rerank(user))
        c = [ranks[t["i"]] - user["base_ranks"][t["i"]] for t in user["targets"] if t["i"] in user["cand_set"]]
        out.append(np.mean(c) if c else np.nan)
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


def table(rows):
    """rows: {column name: {metric: value}} -> markdown lines."""
    keys = list(next(iter(rows.values())).keys())
    lines = ["| metric | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(fmt(rows[r][key]) for r in rows) + " |")
    return lines


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
    ap.add_argument("--profiles", default=None, help="item profiles file (default data/item_profiles/item_profiles.jsonl)")
    ap.add_argument("--no-profiles", action="store_true", help="ignore the item profiles even if the file exists")
    ap.add_argument("--no-swap", action="store_true", help="skip the swapped-persona control table")
    ap.add_argument("--show", type=int, default=1, help="print this many example users")
    ap.add_argument("--show-with", default="R1_like", help="the reranker whose list the examples show")
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

    rerankers, swapped = dict(RERANKERS), {}
    if users and users[0]["persona"] and not args.no_rerank:
        try:
            import rerank as R
        except ImportError as e:                      # no sentence-transformers here: the LightGCN column only
            print(f"persona reranker skipped ({e}); pip install sentence-transformers")
        else:
            R.PARAMS.update(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            enc = R.Encoder()
            ppath = Path(args.profiles) if args.profiles else R.PROFILES_PATH
            profiles = None
            if ppath.exists() and not args.no_profiles:
                profiles = R.Profiles(enc, ppath)
                print(f"item profiles: {len(profiles.lines)} items in {ppath}")
            else:
                print(f"no item profiles ({ppath} missing): R8-R9 skipped; run prep_items.py and item_profiles.py")
            R.attach(users, items, enc, profiles)
            if args.write_features:
                R.write_features(users, run / "features.jsonl")
            rerankers.update(R.RERANKERS)
            if profiles:
                rerankers.update(R.PROFILE_RERANKERS)
            print(f"features of {sum(len(u['list']) for u in users)} (user, item) rows; "
                  f"alpha {args.alpha} beta {args.beta} gamma {args.gamma}")
            if (run / "personas_train.jsonl").exists() and not args.no_train:
                import train as T
                if not (run / "candidates_train.jsonl").exists():
                    n = T.write_candidates(run, inp, k=len(users[0]["candidates"]))
                    print(f"candidates_train.jsonl: lists for {n} training users from embeddings.npz")
                users_train = build_users(run, inp, "train")
                R.attach(users_train, items, enc, profiles)
                model, info = T.fit(users_train, run)
                rerankers["R7_lgbm"] = T.reranker(model, alpha=args.alpha)
                print(f"LightGBM on {info['train_users']} training users ({info['rows']} rows, {info['target_rows']} targets): "
                      f"{info['trees']} trees, validation mean target rank {info['val_target_rank']} on "
                      f"{info['val_users']} users -> lgbm.txt, lgbm.json")
                top = sorted(info["importance_gain"].items(), key=lambda kv: -kv[1])[:5]
                print("  most used features (gain): " + ", ".join(f"{k} {v:.0f}" for k, v in top))
            if not args.no_swap and len(users) > 1:        # the control: every user gets another user's persona
                users_swap = swap_personas(users)
                R.attach(users_swap, items, enc, profiles)
                by_id = {u["user_id"]: u for u in users_swap}
                swapped = {name: (lambda user, f=f: f(by_id[user["user_id"]]))
                           for name, f in rerankers.items() if name != "lightgcn"}

    rows = {name: evaluate(users, f) for name, f in rerankers.items()}
    lines = table(rows)
    if len(rerankers) > 1:
        lines += ["", "mean rank change of the in-list targets per user, bootstrap over users (95% interval):", "",
                  "| reranker | mean change | 2.5% | 97.5% |", "|---|---|---|---|"]
        for name, f in rerankers.items():
            if name != "lightgcn":
                c = user_changes(users, f)
                d, lo, hi = bootstrap_diff(c, np.zeros(len(c)))
                lines.append(f"| {name} | {d:+.1f} | {lo:+.1f} | {hi:+.1f} |")
    if swapped:
        rows_swap = {"lightgcn": rows["lightgcn"], **{name: evaluate(users, f) for name, f in swapped.items()}}
        lines += ["", "control: the same rerankers with another user's persona (what any persona moves):", ""] + table(rows_swap)
        lines += ["", "| reranker | in-list rank change, own persona | another user's persona |", "|---|---|---|"]
        for name in swapped:
            lines.append(f"| {name} | {rows[name]['in_list_rank_change']:+.1f} | {rows_swap[name]['in_list_rank_change']:+.1f} |")
    out = "\n".join(lines)
    print(out)
    (run / "eval.md").write_text(out + "\n", encoding="utf-8")
    for user in users[:args.show]:
        show_user(user, items, rerankers.get(args.show_with, rerank_lightgcn))


if __name__ == "__main__":
    main()
