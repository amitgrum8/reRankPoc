"""Step 6 (CPU, seconds): the trained reranker, level L1 of BLUEPRINT section 6: LightGBM LambdaMART over the
similarity rows of rerank.py. evaluate.py calls this when the run holds personas for the training users
(personas_train.jsonl, from personas.py --set train) and adds the column R7_lgbm to eval.md.

Training queries: one per training user = LightGCN's top-k list for that user, scored with the embeddings the
run saved (no training again), with the user's targets put in, exactly like the test lists. Labels: liked
target (4-5 stars) 2, disliked target (1-2 stars) 0, everything else 1 (unseen candidates, 3-star targets).
LightGBM's lambdarank objective with the gains 0/1/3 then rewards liked > unseen > disliked.

The model never sees the LightGCN score or rank. Targets enter a list with their own base score, usually below
every candidate, so a model that saw the score would learn "low score = target" and lift disliked targets as
much as liked ones. The base score is added back outside the model: R7 = z(base) + alpha * z(model), the same
shape as the formulas, so R7 against R2/R3 compares a learned reading of the persona with a hand-written one.
(The blueprint's rank-matched sampling is the way to let a model learn the fusion as well; not done here.)

Outputs in <run>/: candidates_train.jsonl, lgbm.txt (the model), lgbm.json (sizes, validation NDCG@10,
feature importance by gain).
"""
import json

import numpy as np

import rerank as R

FEATURES = ["like_max", "like_mean", "like_top3", "dis_max", "dis_mean", "dis_top3", "n_like", "n_dis",
            "rocchio", "naive", "tau", "dis_over_tau", "z_like_max", "z_dis_max", "z_rocchio", "z_naive"]
PARAMS = dict(objective="lambdarank", n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, label_gain=[0, 1, 3], random_state=42,
              verbose=-1)


def write_candidates(run, inp, k, set_="train"):
    """candidates_<set>.jsonl from the run's embeddings.npz: the top-k unseen items per user, and every target's
    score and rank among all unseen items (the format lightgcn.py writes for the test users)."""
    users = json.loads((inp / f"{set_}_users.json").read_text(encoding="utf-8"))["users"]
    z = np.load(run / "embeddings.npz")
    ue, ie = z["user"], z["item"]
    lines = []
    for tu in users:
        s = ue[tu["u"]] @ ie.T
        if tu["seen_items"]:
            s[np.array(tu["seen_items"], dtype=np.int64)] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        targets = [{"i": t["i"], "rating": t["rating"], "ts": t["ts"], "score": round(float(s[t["i"]]), 5),
                    "rank": int((s > s[t["i"]]).sum()) + 1} for t in tu["targets"]]
        lines.append(json.dumps({"user_id": tu["user_id"], "u": tu["u"],
                                 "candidates": [[int(a), round(float(s[a]), 5)] for a in top], "targets": targets}))
    (run / f"candidates_{set_}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(users)


def matrix(user):
    """The feature rows of user["list"], in that order (needs user["feat"] from rerank.attach)."""
    f, tau = user["feat"], user["tau"]
    z = {c: R._z(user, c) for c in ("like_max", "dis_max", "rocchio", "naive")}
    rows = [[r["like_max"], r["like_mean"], r["like_top3"], r["dis_max"], r["dis_mean"], r["dis_top3"],
             r["n_like"], r["n_dis"], r["rocchio"], r["naive"], tau, max(0.0, r["dis_max"] - tau),
             z["like_max"](r["like_max"]), z["dis_max"](r["dis_max"]), z["rocchio"](r["rocchio"]), z["naive"](r["naive"])]
            for r in (f[i] for i in user["list"])]
    return np.array(rows, dtype=np.float32)


def labels(user):
    t = {x["i"]: x["rating"] for x in user["targets"]}
    return np.array([1 if t.get(i) is None or t[i] == 3 else 2 if t[i] >= 4 else 0 for i in user["list"]], dtype=np.int32)


def fit(users, run, val_share=0.15, seed=42):
    """LightGBM over the training users' rows; the last val_share of a shuffled user order is held out for early
    stopping on NDCG@10. Writes lgbm.txt and lgbm.json; returns (model, info)."""
    import lightgbm as lgb
    idx = np.random.default_rng(seed).permutation(len(users))
    n_val = int(len(users) * val_share) if len(users) >= 10 else 0
    val, tr = [users[i] for i in idx[:n_val]], [users[i] for i in idx[n_val:]]

    def stack(us):
        return (np.concatenate([matrix(u) for u in us]), np.concatenate([labels(u) for u in us]),
                [len(u["list"]) for u in us])

    Xt, yt, gt = stack(tr)
    model = lgb.LGBMRanker(**PARAMS)
    kw = {}
    if val:
        Xv, yv, gv = stack(val)
        kw = dict(eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10], callbacks=[lgb.early_stopping(100, verbose=False)])
    model.fit(Xt, yt, group=gt, **kw)
    model.booster_.save_model(str(run / "lgbm.txt"))
    best = model.best_score_.get("valid_0", {}) if val else {}
    info = {"train_users": len(tr), "val_users": len(val), "rows": int(len(yt)), "liked_rows": int((yt == 2).sum()),
            "disliked_rows": int((yt == 0).sum()), "trees": int(model.best_iteration_ or model.n_estimators_),
            "val_ndcg@10": round(float(best["ndcg@10"]), 4) if "ndcg@10" in best else None,
            "importance_gain": dict(zip(FEATURES, [round(float(x), 1) for x in model.booster_.feature_importance("gain")])),
            "params": PARAMS}
    (run / "lgbm.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return model, info


def reranker(model, alpha=1.0):
    """R7_lgbm: z(base) + alpha * z(model score), z over the user's candidates."""
    def rerank(user):
        p = dict(zip(user["list"], model.predict(matrix(user)).tolist()))
        cand = np.array([p[i] for i in user["candidates"]])
        mu, sd = cand.mean(), cand.std() or 1.0
        zb = R._z(user, "base")
        return {i: float(zb(user["feat"][i]["base"]) + alpha * (p[i] - mu) / sd) for i in user["list"]}
    return rerank
