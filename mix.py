"""Step 6b (CPU, seconds): learned linear mixes. evaluate.py fits them on the training users (the same lists and
persona rows as R7_lgbm), adds one column per mix to eval.md, the fitted weights below the tables, and mix.json.

  R10_mix          score = w0*z(LightGCN) + w1*z(like_s1) + ... + w5*z(like_s5) + w6*z(dis_s1) + ... + w10*z(dis_s5)
  R11_mix_crit     the same likes; the dislikes against what reviewers criticized (dis_crit_s1..s5), + no_crit
  R12_mix_reviews  the item's own reviews: like lines against its good (4-5 star) review sentences (like_good_s1..s5),
                   dislike lines against its bad (1-2 star) review sentences (dis_bad_s1..s5), + no_bad, n_good, n_bad

like_sK / dis_sK = cosine between the movie text (title + description) and the user's K-th closest like / dislike
line; dis_crit_sK = each dislike line's closest criticized line of the movie's review profile, K-th best over the
dislike lines; like_good_sK / dis_bad_sK = each like / dislike line's closest sentence of the movie's good / bad
reviews, K-th best over the lines. All sorted per movie, so position K means the same for every user (rerank.py).
z = z-score over the user's candidates, as in the formulas. No hand-set alpha or beta.

Missing values: a movie without criticized lines (R11) or without bad reviews (R12) has no dislike match at all.
Its value is set to the user's average (z = 0) and the indicator no_crit / no_bad = 1 carries the fact itself, so a
dislike weight measures matching, not "this movie has no complaints" (which mostly means a well-loved movie).
n_good / n_bad (number of review sentences) let the model discount the higher maximum that more sentences give.
These item columns are the same with another user's persona, so the swapped-persona control still isolates the
persona's part.

Training: logistic regression (Newton steps) on the training users with at least one target inside LightGCN's
top k. Rows = the k candidates; label 1 = the user went on to review the movie (any rating). Targets outside the
top k stay out: they enter with a score below every candidate and would teach "low LightGCN score = target" (see
train.py). An L2 penalty on every weight except the intercept and LightGCN's, picked from L2_GRID by 5-fold
cross-validation over training users (held-out log-likelihood): a large penalty shrinks the persona weights to 0,
which is LightGCN's own order, so noise in many weights cannot reorder the list. Standard errors come from the
penalized Hessian and are approximate (the rows of one user are not independent); the bootstrap and the
swapped-persona control in eval.md are the real test.
"""
import numpy as np

import rerank as R

_K = range(1, R.K_SORTED + 1)
LIKE = [f"like_s{k}" for k in _K]
MIXES = {"R10_mix": ["base"] + LIKE + [f"dis_s{k}" for k in _K],
         "R11_mix_crit": ["base"] + LIKE + [f"dis_crit_s{k}" for k in _K] + ["no_crit"],
         "R12_mix_reviews": (["base"] + [f"like_good_s{k}" for k in _K] + [f"dis_bad_s{k}" for k in _K]
                             + ["no_bad", "n_good", "n_bad"])}
NEEDS = {"R11_mix_crit": "profiles", "R12_mix_reviews": "reviews"}       # mixes that need item profiles / reviews
L2_GRID = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
FOLDS = 5


def matrix(user, ids, inputs):
    """The inputs of the items in ids (rows in that order), each z-scored over the user's candidates; a missing
    value (NaN) and a constant column become 0, the average."""
    f = user["feat"]
    X = np.array([[f[i][c] for c in inputs] for i in ids], dtype=float)
    C = np.array([[f[i][c] for c in inputs] for i in user["candidates"]], dtype=float)
    seen = ~np.isnan(C)
    n = seen.sum(0)
    mu = np.where(seen, C, 0.0).sum(0) / np.maximum(n, 1)
    sd = np.sqrt((np.where(seen, C - mu, 0.0) ** 2).sum(0) / np.maximum(n, 1))
    ok = sd > 1e-9
    Z = np.where(ok, (X - mu) / np.where(ok, sd, 1.0), 0.0)
    return np.where(np.isnan(Z), 0.0, Z)


def _newton(X1, y, pen, max_iter=50):
    """Penalized logistic regression; returns (weights, Newton steps, converged)."""
    w = np.zeros(X1.shape[1])
    w[0] = np.log(y.mean() / (1 - y.mean()))
    for it in range(1, max_iter + 1):
        p = 1.0 / (1.0 + np.exp(-(X1 @ w)))
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + pen
        step = np.linalg.solve(H, X1.T @ (y - p) - pen @ w)
        w += step
        if np.abs(step).max() < 1e-7:
            return w, it, True
    return w, max_iter, False


def _loglik(X1, y, w):
    z = X1 @ w
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


def fit(users, inputs, grid=L2_GRID, folds=FOLDS, seed=42):
    """Logistic regression over the candidates of the users with an in-list target, the L2 penalty picked by
    cross-validation over users. Returns ({input: weight}, info for mix.json)."""
    Xs, ys = [], []
    for user in users:
        t = {x["i"] for x in user["targets"]} & user["cand_set"]
        if t:
            Xs.append(matrix(user, user["candidates"], inputs))
            ys.append(np.array([i in t for i in user["candidates"]], dtype=float))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    X1 = np.hstack([np.ones((len(X), 1)), X])                # column 0 = intercept
    free = [0] + [1 + inputs.index("base")] if "base" in inputs else [0]   # not penalized
    pen_of = lambda l2: np.diag([0.0 if j in free else l2 for j in range(X1.shape[1])])
    fold = np.empty(len(Xs), dtype=int)
    fold[np.random.default_rng(seed).permutation(len(Xs))] = np.arange(len(Xs)) % folds
    row_fold = np.repeat(fold, [len(v) for v in ys])
    cv = {}
    for l2 in grid:
        cv[l2] = 0.0
        for f in range(folds):
            tr = row_fold != f
            cv[l2] += _loglik(X1[~tr], y[~tr], _newton(X1[tr], y[tr], pen_of(l2))[0])
    l2 = max(cv, key=cv.get)
    pen = pen_of(l2)
    w, it, converged = _newton(X1, y, pen)
    p = 1.0 / (1.0 + np.exp(-(X1 @ w)))
    cov = np.linalg.inv((X1 * (p * (1 - p))[:, None]).T @ X1 + pen)

    def total(names):
        idx = [1 + inputs.index(c) for c in names]
        return {"w": round(float(w[idx].sum()), 4), "se": round(float(np.sqrt(cov[np.ix_(idx, idx)].sum())), 4)}

    weights = {c: float(w[1 + k]) for k, c in enumerate(inputs)}
    info = {"train_users": len(users), "users": len(Xs), "rows": int(len(y)), "targets": int(y.sum()), "l2": l2,
            "cv_loglik": {f"{k:g}": round(v, 2) for k, v in cv.items()},
            "iterations": it, "converged": converged, "intercept": round(float(w[0]), 4),
            "weights": {c: {"w": round(weights[c], 4), "se": round(float(np.sqrt(cov[1 + k, 1 + k])), 4)}
                        for k, c in enumerate(inputs)},
            "like_sum": total([c for c in inputs if c.startswith("like_")]),
            "dis_sum": total([c for c in inputs if c.startswith("dis_")])}
    return weights, info


def reranker(weights):
    """A fitted mix: the weights over the z-scored inputs (the intercept does not change the order)."""
    inputs, coef = list(weights), np.array(list(weights.values()))

    def rerank(user):
        return dict(zip(user["list"], (matrix(user, user["list"], inputs) @ coef).tolist()))
    return rerank


def weight_lines(name, info):
    """The weight table of one mix for eval.md."""
    lines = ["", f"{name} weights: logistic regression on {info['users']} training users with a target inside the "
             f"list ({info['targets']} targets, {info['rows']} rows); inputs z-scored within each user's list; "
             f"L2 penalty {info['l2']:g}, picked by {FOLDS}-fold cross-validation over users from "
             f"{', '.join(info['cv_loglik'])}:", "",
             "| input | weight | std. error |", "|---|---|---|"]
    lines += [f"| {c} | {v['w']:+.3f} | {v['se']:.3f} |" for c, v in info["weights"].items()]
    lines += [f"| sum of the like weights | {info['like_sum']['w']:+.3f} | {info['like_sum']['se']:.3f} |",
              f"| sum of the dislike weights | {info['dis_sum']['w']:+.3f} | {info['dis_sum']['se']:.3f} |"]
    return lines
