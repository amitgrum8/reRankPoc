"""Step 4 (local, CPU): metrics of every reranker on the test users -> <run>/eval.md.

  python evaluate.py run                       # run/ holds candidates.jsonl (+ personas.jsonl)
  python evaluate.py run --input data/rr_input --show 2 --show-with R1_like

For every test user, the list = LightGCN's top k (500) + the user's targets (their first reviews after the
cutoff) put into it. A reranker gives every item in that list a score; each target's rank is its place
among the k candidates (other targets don't count).

The main question is how much the reranker moved the user's interacted items (targets, any rating) inside
the list of k candidates:
  in_list_*       targets that LightGCN put in its top k: their mean rank under LightGCN and under the reranker,
                  mean rank change (negative = moved up), and how many moved up / stayed / moved down
  in_list_top10   how many of them end in the top 10
  out_of_list     targets below LightGCN's top k (they enter at rank k+1 and can only move up);
                  out_pulled_in = how many the reranker lifted above at least one candidate
  all_mean_rank   mean rank of every target
  in_list_change_liked / _disliked   the mean rank change of the in-list targets rated 4-5 / 1-2 stars
A second table gives every reranker's mean in-list rank change per user, with a 95% interval from a bootstrap
over users (0 = no change). A third table is the control: the same rerankers with every user given another
user's persona (a fixed derangement). A persona that is not the user's own should not move their items; what
it still moves is the lift any text gives, not personalization.

Standard metrics over all targets: Recall@10/20/50 and NDCG@10/20 (binary relevance) per user, the list being
the k candidates in the reranker's order; a target outside the base list is a miss (the reranker only reorders
the k candidates). Each reranker against the base ranker, paired per user: mean difference, 95% bootstrap
interval over users, and a two-sided sign-flip (randomization) test. Then the paired comparisons that answer the
questions: own persona vs another user's persona, R13_near vs R12, each ablation vs R12. Not corrected for
multiple comparisons.

Every target's rank under every reranker (own and swapped persona) goes to ranks.jsonl.gz, one line per test
user, so any metric can be computed later without the encoder: {"user_id", "u", "n_candidates", "targets": [{"i",
"rating", "in_list", "full_rank" (the base ranker's rank over all unseen items), "rank": {reranker: rank},
"swap": {reranker: rank}}]}; a rank above n_candidates means the target stayed below every candidate.

A reranker is a function rerank(user) -> {item: score} over user["list"] (see RERANKERS at the bottom).
"lightgcn" keeps LightGCN's order; the formulas R1-R5 come from rerank.py (similarity features between the
persona lines and the item text) when sentence-transformers is installed, R8-R9 when item profiles exist
(data/item_profiles/item_profiles.jsonl from item_profiles.py); R7_lgbm from train.py and the learned linear
mixes R10-R12 from mix.py when the run holds personas for the training users (personas_train.jsonl); their fitted
weights go below the tables. R12 compares the like lines with each movie's good review sentences and the dislike
lines with its bad ones (data/rr_input/item_reviews.jsonl.gz). R13_near (and its control R13_near_ctrl) refits
R12 per user with the training users weighted by persona closeness; A1-A6 are the ablations of the dislike part
(mix.py). --encoder picks the text encoder for everything (default bge-small); --tag NAME writes eval_NAME.md,
mix_NAME.json and ranks_NAME.jsonl.gz, so runs with different encoders sit side by side. The first line of eval.md
records the run, encoder and weights.
"""
import argparse
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECALL_AT, NDCG_AT = (10, 20, 50), (10, 20)
PAIRED = ("in_list_change", "ndcg@10", "recall@20")     # the columns of the paired-comparison table


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


def build_users(run, inp, set_="test", limit=None):
    """The users of a set (test or train) that have a persona, with their list, base scores and targets; the first
    `limit` of them when given."""
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
        if limit and len(users) >= limit:
            break
    return users


def donors(users, seed):
    """A fixed derangement: user_id -> another user of the list."""
    order = np.random.default_rng(seed).permutation(len(users))
    return {users[order[k]]["user_id"]: users[order[(k + 1) % len(order)]] for k in range(len(order))}


def swap_personas(users, seed=42):
    """Copies of the users where each one carries another user's persona (a fixed derangement)."""
    donor = donors(users, seed)
    return [{**u, "persona": donor[u["user_id"]]["persona"]} for u in users]


def ranks_of(users, rerank):
    """Every user's target ranks under a reranker (one call of rerank per user)."""
    return [target_ranks(user, rerank(user)) for user in users]


def evaluate(users, rerank=None, k=10, ranks=None):
    """Movement of the user's interacted items inside the candidate list, against LightGCN's order."""
    ranks = ranks if ranks is not None else ranks_of(users, rerank)
    in_base, in_new, chg, all_new, out_n, pulled, liked, disliked = [], [], [], [], 0, 0, [], []
    for user, rk in zip(users, ranks):
        n = len(user["candidates"])
        for t in user["targets"]:
            r0, r1 = user["base_ranks"][t["i"]], rk[t["i"]]
            all_new.append(r1)
            if t["i"] in user["cand_set"]:
                in_base.append(r0); in_new.append(r1); chg.append(r1 - r0)
                if t["rating"] >= 4:
                    liked.append(r1 - r0)
                elif t["rating"] <= 2:
                    disliked.append(r1 - r0)
            else:
                out_n += 1; pulled += r1 <= n
    m = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {"users": len(users), "list_size": len(users[0]["candidates"]) if users else 0,
            "targets": len(all_new), "in_list": len(in_base),
            "in_list_mean_rank_lightgcn": m(in_base), "in_list_mean_rank": m(in_new), "in_list_rank_change": m(chg),
            "in_list_moved_up": sum(c < 0 for c in chg), "in_list_stayed": sum(c == 0 for c in chg),
            "in_list_moved_down": sum(c > 0 for c in chg), f"in_list_top{k}": sum(r <= k for r in in_new),
            "out_of_list": out_n, "out_pulled_in": pulled, "all_mean_rank": m(all_new),
            "in_list_change_liked": m(liked), "in_list_change_disliked": m(disliked)}


def user_changes(users, ranks):
    """Per-user mean rank change of the in-list targets, aligned with users (nan without in-list targets)."""
    out = []
    for user, rk in zip(users, ranks):
        c = [rk[t["i"]] - user["base_ranks"][t["i"]] for t in user["targets"] if t["i"] in user["cand_set"]]
        out.append(np.mean(c) if c else np.nan)
    return np.array(out)


def standard(users, ranks):
    """Per-user Recall@K and NDCG@K over all the user's targets (binary relevance); the list = the k candidates in
    the reranker's order, a target outside the base list is a miss. Plus the per-user in-list rank change."""
    out = {f"recall@{k}": [] for k in RECALL_AT} | {f"ndcg@{k}": [] for k in NDCG_AT}
    for user, rk in zip(users, ranks):
        pos = [rk[t["i"]] for t in user["targets"] if t["i"] in user["cand_set"]]
        n = len(user["targets"])
        for k in RECALL_AT:
            out[f"recall@{k}"].append(sum(p <= k for p in pos) / n)
        for k in NDCG_AT:
            dcg = sum(1.0 / math.log2(p + 1) for p in pos if p <= k)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(n, k) + 1))
            out[f"ndcg@{k}"].append(dcg / idcg)
    out = {key: np.array(v) for key, v in out.items()}
    out["in_list_change"] = user_changes(users, ranks)
    return out


def bootstrap_diff(a, b, n=2000, seed=42):
    """Mean of a - b over users, with a 95% interval from a paired bootstrap over users."""
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a, b, n_boot=2000, n_perm=10000, seed=42):
    """Paired comparison over users (nan = the user has no value, dropped): mean of a - b, 95% bootstrap interval,
    and the two-sided sign-flip test's p."""
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d))) @ d / len(d)
    p = (1 + int((np.abs(flips) >= abs(d.mean()) - 1e-12).sum())) / (n_perm + 1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), p


def cell(res, scale=1.0, digits=1):
    """'-3.1 [-5.0, -1.2] p 0.002' for a paired result."""
    d, lo, hi, p = res
    if math.isnan(d):
        return "n/a"
    f = lambda x: f"{x * scale:+.{digits}f}"
    return f"{f(d)} [{f(lo)}, {f(hi)}] " + ("p<0.001" if p < 0.001 else f"p {p:.3f}")


def table(rows, fmt=None):
    """rows: {column name: {metric: value}} -> markdown lines."""
    keys = list(next(iter(rows.values())).keys())
    lines = ["| metric | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    fmt = fmt or (lambda v: f"{v:.1f}" if isinstance(v, float) else str(v))
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(fmt(rows[r][key]) for r in rows) + " |")
    return lines


def write_ranks(path, users, names, ranks, ranks_swap):
    """Every target's rank under every reranker, own and swapped persona: one line per test user."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for k, user in enumerate(users):
            tg = [{"i": t["i"], "rating": t["rating"], "in_list": t["i"] in user["cand_set"], "full_rank": t.get("rank"),
                   "rank": {n: ranks[n][k][t["i"]] for n in names},
                   "swap": {n: ranks_swap[n][k][t["i"]] for n in ranks_swap}} for t in user["targets"]]
            f.write(json.dumps({"user_id": user["user_id"], "u": user["u"], "n_candidates": len(user["candidates"]),
                                "targets": tg}) + "\n")


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
    ap.add_argument("--no-train", action="store_true",
                    help="skip the trained rerankers (R7_lgbm, the mixes, R13, the ablations) even if "
                         "personas_train.jsonl exists")
    ap.add_argument("--no-lgbm", action="store_true", help="skip R7_lgbm (leaves lgbm.txt / lgbm.json untouched)")
    ap.add_argument("--no-near", action="store_true", help="skip R13_near and its control")
    ap.add_argument("--no-ablations", action="store_true", help="skip the ablations A1-A6 of the dislike part")
    ap.add_argument("--users", type=int, default=None, help="only the first N test users (smoke test)")
    ap.add_argument("--train-users", type=int, default=None, help="only the first N training users (smoke test)")
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5",
                    help="sentence-transformers model for every text; rerank.ENCODERS lists the known ones "
                         "(e.g. BAAI/bge-large-en-v1.5, Qwen/Qwen3-Embedding-4B)")
    ap.add_argument("--no-reviews", action="store_true",
                    help="skip the items' own review sentences (R12) even if item_reviews.jsonl.gz exists")
    ap.add_argument("--tag", default="",
                    help="write eval_<tag>.md, mix_<tag>.json, ranks_<tag>.jsonl.gz instead of eval.md, mix.json, ranks.jsonl.gz")
    ap.add_argument("--write-features", action="store_true", help="write the test users' feature rows to features.jsonl")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the like similarity / model score (rerank.py)")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the dislike penalty (rerank.py)")
    ap.add_argument("--gamma", type=float, default=1.0, help="dislike weight inside the Rocchio vector (rerank.py)")
    args = ap.parse_args()
    run, inp = Path(args.run), Path(args.input)
    sfx = f"_{args.tag}" if args.tag else ""
    t0 = time.time()
    users = build_users(run, inp, limit=args.users)
    items = json.loads((inp / "items.json").read_text(encoding="utf-8"))
    print(f"{len(users)} test users" + (" (with a persona)" if users and users[0]["persona"] else "")
          + (f", lists of {len(users[0]['candidates'])}" if users else ""))

    rerankers, swapped, mix_info, near = dict(RERANKERS), {}, {}, None
    if users and users[0]["persona"] and not args.no_rerank:
        try:
            import rerank as R
        except ImportError as e:                      # no sentence-transformers here: the LightGCN column only
            print(f"persona reranker skipped ({e}); pip install sentence-transformers")
        else:
            R.PARAMS.update(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
            enc = R.Encoder(args.encoder)
            ppath = Path(args.profiles) if args.profiles else R.PROFILES_PATH
            profiles = None
            if ppath.exists() and not args.no_profiles:
                profiles = R.Profiles(enc, ppath)
                print(f"item profiles: {len(profiles.lines)} items in {ppath}")
            else:
                print(f"no item profiles ({ppath} missing): R8-R9 skipped; run prep_items.py and item_profiles.py")
            rpath, reviews = inp / "item_reviews.jsonl.gz", None
            if rpath.exists() and not args.no_reviews:
                reviews = R.Reviews(enc, rpath)
                print(f"item reviews: {len(reviews.units)} items in {rpath}")
            else:
                print(f"no item reviews ({rpath} missing or --no-reviews): R12, R13 and the ablations skipped")
            R.attach(users, items, enc, profiles, reviews)
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
                users_train = build_users(run, inp, "train", limit=args.train_users)
                R.attach(users_train, items, enc, profiles, reviews)
                if not args.no_lgbm:
                    model, info = T.fit(users_train, run)
                    rerankers["R7_lgbm"] = T.reranker(model, alpha=args.alpha)
                    print(f"LightGBM on {info['train_users']} training users ({info['rows']} rows, {info['target_rows']} targets): "
                          f"{info['trees']} trees, validation mean target rank {info['val_target_rank']} on "
                          f"{info['val_users']} users -> lgbm.txt, lgbm.json")
                    top = sorted(info["importance_gain"].items(), key=lambda kv: -kv[1])[:5]
                    print("  most used features (gain): " + ", ".join(f"{k} {v:.0f}" for k, v in top))
                import mix as MX

                def fit_mix(name, inputs):
                    need = MX.NEEDS.get(name)
                    if (need == "profiles" and not profiles) or (need == "reviews" and not reviews):
                        return
                    weights, mix_info[name] = MX.fit(users_train, inputs, label=MX.LABELS.get(name, "any"))
                    rerankers[name] = MX.reranker(weights)
                    fit_check = evaluate(users_train, rerankers[name])
                    m = mix_info[name]
                    print(f"{name} on {m['users']} training users with an in-list target ({m['targets']} targets, "
                          f"{m['iterations']} Newton steps, converged {m['converged']}): in-list rank change on the "
                          f"training users {fit_check['in_list_rank_change']:+.1f} (in-sample)", flush=True)

                for name, inputs in MX.MIXES.items():
                    fit_mix(name, inputs)
                if not args.no_near and MX.REFERENCE in mix_info:
                    t1 = time.time()
                    near = MX.NearMix(users_train, MX.MIXES[MX.REFERENCE], mix_info[MX.REFERENCE]["l2"])
                    print(f"R13_near: leave-one-out over {len(near.users)} training users, alphas "
                          f"{', '.join(f'{a:g}' for a in near.alphas)} ...", flush=True)
                    alpha = near.select()
                    print(f"R13_near: alpha {alpha:g} picked ({(time.time() - t1) / 60:.1f} min)", flush=True)
                    rerankers["R13_near"] = near.reranker("own")
                    ctrl = donors(users, seed=7)        # closeness from another test user's persona
                    rerankers["R13_near_ctrl"] = near.reranker("ctrl", pv_of=lambda u: ctrl[u["user_id"]]["pv"])
                if not args.no_ablations:
                    for name, inputs in MX.ABLATIONS.items():
                        fit_mix(name, inputs)
            if not args.no_swap and len(users) > 1:        # the control: every user gets another user's persona
                users_swap = swap_personas(users)
                R.attach(users_swap, items, enc, profiles, reviews)
                by_id = {u["user_id"]: u for u in users_swap}
                own = {"R13_near": near.reranker("swap")} if near is not None else {}   # its own fit summary
                swapped = {name: (lambda user, f=own.get(name, f): f(by_id[user["user_id"]]))
                           for name, f in rerankers.items() if name != "lightgcn" and not name.endswith("_ctrl")}

    names = list(rerankers)
    print(f"scoring {len(names)} rerankers" + (f" + {len(swapped)} with swapped personas" if swapped else "")
          + " ...", flush=True)
    ranks = {name: ranks_of(users, f) for name, f in rerankers.items()}
    ranks_swap = {name: ranks_of(users, f) for name, f in swapped.items()}
    rows = {name: evaluate(users, ranks=ranks[name]) for name in names}
    lines = [f"run {run.name} | encoder {args.encoder if len(rerankers) > 1 else 'none'} | alpha {args.alpha} "
             f"beta {args.beta} gamma {args.gamma}"
             + (f" | first {args.users} test users" if args.users else "")
             + (f" | first {args.train_users} training users" if args.train_users else ""), ""] + table(rows)
    if len(rerankers) > 1:
        lines += ["", "mean rank change of the in-list targets per user, bootstrap over users (95% interval):", "",
                  "| reranker | mean change | 2.5% | 97.5% |", "|---|---|---|---|"]
        for name in names:
            if name != "lightgcn":
                c = user_changes(users, ranks[name])
                d, lo, hi = bootstrap_diff(c, np.zeros(len(c)))
                lines.append(f"| {name} | {d:+.1f} | {lo:+.1f} | {hi:+.1f} |")
    rows_swap = {}
    if swapped:
        rows_swap = {"lightgcn": rows["lightgcn"], **{name: evaluate(users, ranks=ranks_swap[name]) for name in swapped}}
        lines += ["", "control: the same rerankers with another user's persona (what any persona moves):", ""] + table(rows_swap)
        lines += ["", "| reranker | in-list rank change, own persona | another user's persona |", "|---|---|---|"]
        for name in swapped:
            lines.append(f"| {name} | {rows[name]['in_list_rank_change']:+.1f} | {rows_swap[name]['in_list_rank_change']:+.1f} |")

    # standard metrics over all targets, and the paired tests
    std = {name: standard(users, ranks[name]) for name in names}
    std_swap = {name: standard(users, ranks_swap[name]) for name in swapped}
    metrics = [f"recall@{k}" for k in RECALL_AT] + [f"ndcg@{k}" for k in NDCG_AT]
    mean_of = lambda s: {m: float(np.mean(s[m])) for m in metrics}
    fmt4 = lambda v: f"{v:.4f}"
    lines += ["", "## Standard metrics over all targets", "",
              f"Per user, averaged over the {len(users)} users; the list = the {len(users[0]['candidates'])} candidates "
              "in the reranker's order, a target outside the base list is a miss. Own persona:", ""]
    lines += table({name: mean_of(std[name]) for name in names}, fmt4)
    if swapped:
        lines += ["", "the same with another user's persona:", ""]
        lines += table({"lightgcn": mean_of(std["lightgcn"]), **{name: mean_of(std_swap[name]) for name in swapped}}, fmt4)
    if len(names) > 1:
        lines += ["", "each reranker minus the base ranker, paired per user (difference x100, 95% bootstrap interval, "
                  "two-sided sign-flip p):", "",
                  "| reranker | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
        for name in names[1:]:
            lines.append(f"| {name} | " + " | ".join(cell(paired(std[name][m], std["lightgcn"][m]), 100, 2)
                                                    for m in metrics) + " |")
        comps = [(f"{n}: own - another user's persona", std[n], std_swap[n]) for n in swapped]
        ref = MX.REFERENCE if mix_info else None
        if ref in std:
            if "R13_near" in std:
                comps += [(f"R13_near - {ref}", std["R13_near"], std[ref]),
                          (f"R13_near_ctrl - {ref}", std["R13_near_ctrl"], std[ref]),
                          ("R13_near - R13_near_ctrl", std["R13_near"], std["R13_near_ctrl"])]
            comps += [(f"{a} - {ref}", std[a], std[ref]) for a in MX.ABLATIONS if a in std]
        if comps:
            lines += ["", "## Paired comparisons per user", "",
                      "a - b per user: mean, 95% bootstrap interval, two-sided sign-flip p. In-list rank change: "
                      "negative = a moves the targets higher than b; ndcg and recall x100: positive = a is better.", "",
                      "| comparison | " + " | ".join(PAIRED) + " |", "|---|" + "---|" * len(PAIRED)]
            for label, a, b in comps:
                lines.append(f"| {label} | " + " | ".join(
                    cell(paired(a[m], b[m]), 1 if m == "in_list_change" else 100, 1 if m == "in_list_change" else 2)
                    for m in PAIRED) + " |")
    if near is not None:
        mix_info["R13_near"] = near.info()
    for name, m in mix_info.items():
        lines += MX.near_lines(name, m) if name == "R13_near" else MX.weight_lines(name, m)
    out = "\n".join(lines)
    print(out)
    (run / f"eval{sfx}.md").write_text(out + "\n", encoding="utf-8")
    if mix_info:
        (run / f"mix{sfx}.json").write_text(json.dumps(mix_info, indent=1), encoding="utf-8")
    write_ranks(run / f"ranks{sfx}.jsonl.gz", users, names, ranks, ranks_swap)
    print(f"\n-> {run / f'eval{sfx}.md'}, {run / f'ranks{sfx}.jsonl.gz'}"
          + (f", {run / f'mix{sfx}.json'}" if mix_info else "") + f"  ({(time.time() - t0) / 60:.1f} min)")
    for user in users[:args.show]:
        show_user(user, items, rerankers.get(args.show_with, rerank_lightgcn))


if __name__ == "__main__":
    main()
